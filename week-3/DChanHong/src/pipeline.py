"""
Indexing 파이프라인 모듈
PDF 로드 → 청킹 → 임베딩 → FAISS 저장까지의 전체 프로세스를 제어합니다.
"""

from typing import Optional

from src.core.loader import load_pdf, load_pdfs_from_directory
from src.core.splitter import split_documents
from src.core.embedder import create_vector_store
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_indexing_pipeline(
    source: str,
    is_directory: bool = False,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> None:
    """
    전체 인덱싱 파이프라인을 실행합니다.

    순서: PDF 로드 → 청킹 → 임베딩 → FAISS 저장

    Args:
        source: PDF 파일 경로 또는 디렉토리 경로
        is_directory: True이면 디렉토리 내 전체 PDF 로드
        chunk_size: 청크 크기 (None이면 설정값 사용)
        chunk_overlap: 청크 오버랩 크기 (None이면 설정값 사용)
    """
    logger.info("=" * 60)
    logger.info("인덱싱 파이프라인 시작")
    logger.info("=" * 60)

    # 1단계: PDF 로드
    logger.info("[1/3] PDF 로드 중...")
    if is_directory:
        documents = load_pdfs_from_directory(source)
    else:
        documents = load_pdf(source)

    if not documents:
        logger.warning("로드된 문서가 없습니다. 파이프라인을 종료합니다.")
        return

    # 2단계: 텍스트 분할
    logger.info("[2/3] 텍스트 청킹 중...")
    split_kwargs = {}
    if chunk_size is not None:
        split_kwargs["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        split_kwargs["chunk_overlap"] = chunk_overlap

    chunks = split_documents(documents, **split_kwargs)

    # 3단계: 임베딩 및 저장
    logger.info("[3/3] 임베딩 생성 및 Chroma 벡터 DB 저장 중...")
    
    # Chroma는 create_vector_store 안에서 persist_directory 설정 시 자동 저장됩니다.
    create_vector_store(chunks)

    logger.info("=" * 60)
    logger.info("인덱싱 파이프라인 완료")
    logger.info(f"  - 로드된 문서: {len(documents)}개")
    logger.info(f"  - 생성된 청크: {len(chunks)}개")
    logger.info("=" * 60)
