from __future__ import annotations as _annotations

import asyncio
import json
import supabase
from collections.abc import AsyncIterator
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, TypeVar
import uuid

import fastapi
import logfire
from supabase import create_client
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from typing_extensions import LiteralString, ParamSpec, TypedDict

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from config import Settings

new_user_id = '3fb73da7-6038-452f-95ae-e907159dee42'

THIS_DIR = Path(__file__).parent

settings = Settings()
supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
app = fastapi.FastAPI()
logfire.instrument_fastapi(app)
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'llama-3.3-70b',
    provider=OpenAIProvider(base_url="https://api.cerebras.ai/v1", api_key=settings.CEREBRAS_API_KEY),
)

syspr = '''
    You are a Helpful chatbot that will answer questions in detail and will not stray away from the topic at hand.
'''

agent = Agent(
    model,
    system_prompt=syspr
    )

async def get_db(request: Request) -> Database:
    user_id = new_user_id
    return Database(user_id=user_id)

@app.get('/')
async def index() -> FileResponse:
    return FileResponse((THIS_DIR / 'chat_app.html'), media_type='text/html')


@app.get('/chat_app.ts')
async def main_ts() -> FileResponse:
    """Get the raw typescript code, it's compiled in the browser, forgive me."""
    return FileResponse((THIS_DIR / 'chat_app.ts'), media_type='text/plain')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': first_part.content,
            }
    raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')


@app.post('/chat/')
async def post_chat(
    prompt: Annotated[str, fastapi.Form()], database: Database = Depends(get_db)
) -> StreamingResponse:
    async def stream_messages():
        """Streams new line delimited JSON `Message`s to the client."""
        # stream the user prompt so that can be displayed straight away
        yield (
            json.dumps(
                {
                    'role': 'user',
                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                    'content': prompt,
                }
            ).encode('utf-8')
            + b'\n'
        )
        # get the chat history so far to pass as context to the agent
        messages = await database.get_messages()
        # run the agent with the user prompt and the chat history
        async with agent.run_stream(prompt, message_history=messages) as result:
            async for text in result.stream(debounce_by=0.01):
                # text here is a `str` and the frontend wants
                # JSON encoded ModelResponse, so we create one
                m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                yield json.dumps(to_chat_message(m)).encode('utf-8') + b'\n'

        # add new messages (e.g. the user prompt and the agent response in this case) to the database
        await database.add_messages(result.new_messages_json())

    return StreamingResponse(stream_messages(), media_type='text/plain')


P = ParamSpec('P')
R = TypeVar('R')



@dataclass
class Database:
    user_id: str
    window_size: int = 5

    async def add_messages(self, messages: bytes):
        await asyncio.to_thread(
            lambda: supabase.table("messages")
            .insert({"user_id": self.user_id, "message_list":json.loads(messages.decode("utf-8"))})
            .execute()
        )

    async def get_messages(self) -> list[ModelMessage]:
        resp = await asyncio.to_thread(
            lambda: supabase
                .table("messages")
                .select("message_list")
                .eq("user_id", self.user_id)
                .order("created_at", desc=True)
                .limit(self.window_size * 2)
                .execute()
        )
        rows = resp.data or []
        messages: list[ModelMessage] = []
        for row in reversed(resp.data or []):
            messages.extend(ModelMessagesTypeAdapter.validate_python(row["message_list"]))
        return messages[-self.window_size:]



if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'pydantic_ai_examples.chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)]
    )


# def main():
#     print("🤖 Terminal Chatbot (type 'exit' to quit)")
#     while True:
#         user_input = input("You: ")
#         if user_input.strip().lower() in ["exit", "quit"]:
#             print("Bot: Goodbye!")
#             break

#         # Get response from agent using .chat()
#         try:
#             response = asyncio.run(agent.run(user_input))
#             print(f"Bot: {response.output}")
#         except Exception as e:
#             print(f"⚠️ Error: {e}")

# if __name__ == "__main__":
#     main()
