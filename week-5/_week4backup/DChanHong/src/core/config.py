import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# 현재 파일(config.py)에서 2단계 위로 가면 DChanHong 폴더임
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

# .env 파일이 있으면 시스템 환경 변수보다 우선하여 로드 (override=True)
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    load_dotenv(override=True)

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    COHERE_API_KEY: str = ""

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gemini-3-flash-preview"

    # PDF Data path (DChanHong 폴더 기준 상대 경로)
    DATA_PATH: str = "data"
    # ChromaDB storage path
    STORAGE_PATH: str = "./storage"

    # env_file을 여러 경로 후보로 지정하여 더 견고하게 함
    model_config = SettingsConfigDict(
        env_file=(str(ENV_PATH), ".env"), 
        extra="ignore"
    )

settings = Settings()

# 디버그용 출력: .env 경로 및 키 확인
print(f"DEBUG: BASE_DIR: {BASE_DIR}")
print(f"DEBUG: Using .env file at: {ENV_PATH}")
if os.path.exists(ENV_PATH):
    print(f"DEBUG: .env file found at ENV_PATH.")
elif os.path.exists(".env"):
    print(f"DEBUG: .env file found in Current Working Directory.")
else:
    print(f"DEBUG: .env file NOT found anywhere!")

def mask_key(key):
    if not key: return "None"
    return f"{key[:7]}...{key[-4:]}"

print(f"DEBUG: GEMINI_API_KEY: {mask_key(settings.GEMINI_API_KEY)}")
print(f"DEBUG: OPENAI_API_KEY: {mask_key(settings.OPENAI_API_KEY)}")
print(f"DEBUG: COHERE_API_KEY: {mask_key(settings.COHERE_API_KEY)}")
