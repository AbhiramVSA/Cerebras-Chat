from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from config import Settings
import asyncio

settings = Settings()

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





