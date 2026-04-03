"""
임베딩 & Chroma 인터페이스 모듈
텍스트 임베딩을 생성하고 Chroma 벡터 스토어를 관리합니다.
"""

# List, Optional: 파이썬의 타입 힌팅(타입 명시)을 위한 내장 모듈입니다.
from typing import List, Optional

# OpenAIEmbeddings: 텍스트를 숫자의 배열(벡터)로 변환해주는 OpenAI의 임베딩 모델을 사용하기 위한 LangChain 모듈입니다.
from langchain_openai import OpenAIEmbeddings
# Chroma: 생성된 텍스트 벡터들을 로컬에 저장하고 검색할 수 있게 해주는 ChromaDB의 LangChain 통합 모듈입니다.
from langchain_chroma import Chroma
# Document: LangChain의 핵심 객체로, 문서의 텍스트 데이터와 메타데이터(페이지, 출처 등)를 함께 담는 클래스입니다.
from langchain_core.documents import Document

# 설정 파일 및 로거: 직접 정의된 키값 설정 및 로그 기록용 함수들을 가져옵니다.
from src.config.settings import OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_DB_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI 임베딩 모델 인스턴스를 반환합니다."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )


def create_vector_store(documents: List[Document], persist_directory: Optional[str] = None) -> Chroma:
    """
    문서 리스트로부터 Chroma 벡터 스토어를 생성 및 로컬에 저장합니다.

    Args:
        documents: 임베딩할 문서 리스트
        persist_directory: 로컬 DB 저장 경로 (디폴트: config 사용)

    Returns:
        Chroma: 생성된 벡터 스토어 인스턴스
    """
    logger.info(f"벡터 스토어 생성 시작: {len(documents)}개 문서")

    embeddings = get_embeddings()
    save_path = persist_directory or str(CHROMA_DB_DIR)
    
    # Chroma는 persist_directory를 지정하면 자동으로 로컬 db 파일을 생성 및 저장합니다.
    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,
        persist_directory=save_path
    )

    logger.info(f"Chroma 벡터 스토어 생성 및 저장 완료: {save_path}")
    return vector_store


def load_vector_store(persist_directory: Optional[str] = None) -> Chroma:
    """
    로컬에 저장된 Chroma 벡터 스토어를 로드합니다.

    Args:
        persist_directory: 로드 경로 (기본값: settings.CHROMA_DB_DIR)

    Returns:
        Chroma: 로드된 벡터 스토어
    """
    load_path = persist_directory or str(CHROMA_DB_DIR)
    logger.info(f"Chroma 벡터 스토어 로드 시도: {load_path}")

    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=load_path, 
        embedding_function=embeddings
    )

    logger.info("Chroma 벡터 스토어 로드 성공")
    return vector_store
