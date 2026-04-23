import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# metadata-full/ 폴더 기준으로 .env 를 찾는다.
# 이 파일 위치: metadata-full/src/core/config.py
# parents[2] => metadata-full/
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    load_dotenv(override=True)


class Settings(BaseSettings):
    # API keys
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    COHERE_API_KEY: str = ""

    # Models
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gemini-1.5-flash"
    RERANK_MODEL: str = "rerank-multilingual-v3.0"

    # 상대 년도 표현 해석 기준
    REFERENCE_YEAR: int = 2026

    # Paths (metadata-full/ 기준 상대 경로)
    DATA_PATH: str = "data"
    STORAGE_PATH: str = "./storage"

    model_config = SettingsConfigDict(
        env_file=(str(ENV_PATH), ".env"),
        extra="ignore",
    )


settings = Settings()


def _mask(key: str) -> str:
    if not key:
        return "None"
    return f"{key[:6]}...{key[-4:]}" if len(key) > 10 else "***"


print(f"[config] BASE_DIR       : {BASE_DIR}")
print(f"[config] ENV_PATH       : {ENV_PATH} (exists={ENV_PATH.exists()})")
print(f"[config] OPENAI_API_KEY : {_mask(settings.OPENAI_API_KEY)}")
print(f"[config] GEMINI_API_KEY : {_mask(settings.GEMINI_API_KEY)}")
print(f"[config] COHERE_API_KEY : {_mask(settings.COHERE_API_KEY)}")
print(f"[config] REFERENCE_YEAR : {settings.REFERENCE_YEAR}")
