from typing import List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    include_sources: bool = True
    k: int = 10
    candidates: int = 20


class SourceDocument(BaseModel):
    content: str
    metadata: dict


class FilterInfo(BaseModel):
    applied_years: List[str]
    is_cross_year: bool
    rationale: str
    fallback: bool = False


class QueryResponse(BaseModel):
    answer: str
    retrieved_contexts: List[str]
    sources: Optional[List[SourceDocument]] = None
    filter_info: Optional[FilterInfo] = None


class IndexResponse(BaseModel):
    status: str
    files_processed: List[str]
    total_chunks: int
    bm25_indexed: int
