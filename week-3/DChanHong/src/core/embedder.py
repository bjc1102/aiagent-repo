"""
임베딩 & Chroma 인터페이스 모듈
텍스트 임베딩을 생성하고 Chroma 벡터 스토어를 관리합니다.
"""

from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

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
