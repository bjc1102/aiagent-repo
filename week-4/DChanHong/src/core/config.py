import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# DChanHong/.env 경로를 계산 (src/core/config.py 기준 parents[2])
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

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
