import os
from pathlib import Path
from pydantic_settings import BaseSettings

API_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

class Settings(BaseSettings):
    APP_ENV: str = "development"
    DATABASE_URL: str = "sqlite+aiosqlite:///./audio_sentiment.db"
    STORAGE_PATH: str = "./storage/audio"
    MAX_AUDIO_SIZE_MB: int = 100
    ELEVENLABS_API_KEY: str | None = None
    ELEVENLABS_STT_MODEL_ID: str = "scribe_v2"

    class Config:
        env_file = API_ENV_FILE

settings = Settings()
