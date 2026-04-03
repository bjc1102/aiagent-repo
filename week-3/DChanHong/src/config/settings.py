"""
환경 설정 모듈
API 키, 모델명, Chunk Size 등 프로젝트 전반 설정을 관리합니다.
"""

# os: 운영체제와 상호작용하기 위한 유틸리티로, 환경 변수 등을 제어하기 위해 사용하는 파이썬 내장 모듈입니다.
import os
# Path: 파일 및 디렉토리 경로 관련 작업을 편리하게 해주는 객체지향 경로 처리 내장 라이브러리입니다.
from pathlib import Path
# load_dotenv (python-dotenv 라이브러리): '.env' 파일에 저장된 환경 변수를 읽어서 파이썬 안으로 로드해주는 외부 라이브러리입니다.
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
