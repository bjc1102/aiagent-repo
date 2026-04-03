"""
검색(Retriever) 모듈
Chroma DB에서 관련 문서를 검색합니다.
"""

from typing import List
from langchain_core.documents import Document
from src.core.embedder import load_vector_store
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_relevant_documents(query: str, k: int = 5) -> List[Document]:
    """
    질문과 관련된 문서를 Chroma DB에서 검색합니다.
    
    Args:
        query: 검색어
        k: 반환할 문서 수
        
    Returns:
        List[Document]: 관련 문서 리스트
    """
    logger.info(f"검색 시작: '{query}' (k={k})")
    
    try:
        vector_store = load_vector_store()
        # similarity_search를 사용하여 관련 문서를 가져옵니다.
        documents = vector_store.similarity_search(query, k=k)
        
        logger.info(f"검색 완료: {len(documents)}개 문서 발견")
        return documents
        
    except Exception as e:
        logger.error(f"문서 검색 중 오류 발생: {e}")
        return []
