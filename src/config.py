from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=".env", override=True)

class Settings(BaseSettings):
    CEREBRAS_API_KEY: str
    LOGFIRE_KEY: str
    SUPABASE_URL: str
    SUPABASE_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

