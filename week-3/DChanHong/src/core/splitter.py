"""
청킹 전략 모듈
텍스트를 적절한 크기로 분할하며, Table 보존 로직을 포함합니다.
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    문서 리스트를 청크 단위로 분할합니다.

    Args:
        documents: 분할할 문서 리스트
        chunk_size: 청크 크기 (기본값: settings.CHUNK_SIZE)
        chunk_overlap: 청크 간 오버랩 크기 (기본값: settings.CHUNK_OVERLAP)

    Returns:
        List[Document]: 분할된 청크 리스트
    """
    logger.info(
        f"문서 분할 시작 (chunk_size={chunk_size}, overlap={chunk_overlap})"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"문서 분할 완료: {len(documents)}개 문서 → {len(chunks)}개 청크")

    return chunks


def split_documents_with_table_preservation(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    테이블 구조를 보존하면서 문서를 분할합니다.
    테이블이 포함된 청크는 분할하지 않고 유지합니다.

    Args:
        documents: 분할할 문서 리스트
        chunk_size: 청크 크기
        chunk_overlap: 청크 간 오버랩 크기

    Returns:
        List[Document]: 분할된 청크 리스트 (테이블 보존)
    """
    logger.info("테이블 보존 모드로 문서 분할 시작")

    # TODO: 테이블 감지 및 보존 로직 구현
    # 현재는 기본 분할 로직을 사용합니다.
    chunks = split_documents(documents, chunk_size, chunk_overlap)

    return chunks
