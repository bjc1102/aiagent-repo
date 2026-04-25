from typing import List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    include_sources: bool = True
    k: int = 10


class SourceDocument(BaseModel):
    content: str
    metadata: dict


class FilterInfo(BaseModel):
    """적용된 pre-retrieval 필터 내역 (디버그용)"""
    applied_years: List[str]
    is_cross_year: bool
    rationale: str
    fallback: bool = False  # 결과 부족으로 필터를 해제했는지


class QueryResponse(BaseModel):
    answer: str
    retrieved_contexts: List[str]
    sources: Optional[List[SourceDocument]] = None
    filter_info: Optional[FilterInfo] = None


class IndexResponse(BaseModel):
    status: str
    files_processed: List[str]
    total_chunks: int
