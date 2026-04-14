import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# week-4/.env 경로를 계산
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gemini-3-flash-preview"

    # PDF Data path (DChanHong 폴더 기준 상대 경로)
    DATA_PATH: str = "../data"
    # ChromaDB storage path
    STORAGE_PATH: str = "./storage"

    model_config = SettingsConfigDict(env_file=str(ENV_PATH), extra="ignore")

settings = Settings()
