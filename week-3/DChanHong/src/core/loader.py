"""
PDF 로드 및 검증 모듈
PDF 파일을 로드하고, 유효성을 검증합니다.
"""

# Path: 파일 시스템 경로(파일이나 폴더 위치)를 편리하게 다루기 위해 사용하는 파이썬 내장 라이브러리입니다.
from pathlib import Path
# List: 리스트에 들어갈 데이터의 타입을 지정할 때 사용하는 타입 힌팅 모듈입니다.
from typing import List

# PyPDFLoader (langchain-community 모듈): LangChain 커뮤니티에서 제공하는 PDF 로더로, PDF 파일을 페이지 단위로 읽어 Document 객체로 변환해줍니다.
from langchain_community.document_loaders import PyPDFLoader
# Document (langchain-core 모듈): LangChain의 핵심 클래스 중 하나로, 텍스트와 함께 메타데이터(페이지 번호, 소스 경로 등) 등 문서의 정보를 보관하는 객체입니다.
from langchain_core.documents import Document

# 직접 정의한 로그 유틸리티를 가져옵니다.
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_pdf(file_path: str) -> List[Document]:
    """
    PDF 파일을 로드하여 Document 리스트로 반환합니다.

    Args:
        file_path: PDF 파일 경로

    Returns:
        List[Document]: 로드된 문서 리스트

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        ValueError: PDF 파일이 아닐 경우
    """
    path = Path(file_path)

    # 파일 존재 여부 검증
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    # 확장자 검증
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF 파일이 아닙니다: {file_path}")

    logger.info(f"PDF 로드 시작: {file_path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    logger.info(f"PDF 로드 완료: {len(documents)}페이지 로드됨")
    return documents


def load_pdfs_from_directory(directory: str) -> List[Document]:
    """
    디렉토리 내 모든 PDF 파일을 로드합니다.

    Args:
        directory: PDF 파일이 있는 디렉토리 경로

    Returns:
        List[Document]: 로드된 전체 문서 리스트
    """
    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise NotADirectoryError(f"디렉토리가 아닙니다: {directory}")

    pdf_files = list(dir_path.glob("*.pdf"))
    logger.info(f"디렉토리 내 PDF 파일 {len(pdf_files)}개 발견")

    all_documents: List[Document] = []
    for pdf_file in pdf_files:
        try:
            docs = load_pdf(str(pdf_file))
            all_documents.extend(docs)
        except Exception as e:
            logger.error(f"PDF 로드 실패 ({pdf_file}): {e}")

    logger.info(f"총 {len(all_documents)}개 문서 로드 완료")
    return all_documents
