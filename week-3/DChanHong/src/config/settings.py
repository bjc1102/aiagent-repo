"""
환경 설정 모듈
API 키, 모델명, Chunk Size 등 프로젝트 전반 설정을 관리합니다.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# ──────────────────────────────────────────────
# 프로젝트 경로 설정
# ──────────────────────────────────────────────
# settings.py -> config -> src -> project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
LOGS_DIR = BASE_DIR / "logs"

# 디렉토리 자동 생성
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# API 키
# ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ──────────────────────────────────────────────
# 모델 설정
# ──────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ──────────────────────────────────────────────
# 청킹 설정
# ──────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ──────────────────────────────────────────────
# Chroma 설정
# ──────────────────────────────────────────────
CHROMA_DB_DIR = STORAGE_DIR / "chroma_db"
