import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gemini-3-flash-preview"
    
    # PDF Data path
    DATA_PATH: str = "../../data"
    # ChromaDB storage path
    STORAGE_PATH: str = "./storage"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
