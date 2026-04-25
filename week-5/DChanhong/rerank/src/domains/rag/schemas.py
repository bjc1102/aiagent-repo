from typing import List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    include_sources: bool = True
    k: int = 10                     # 최종 LLM 에 넘길 청크 수
    candidates: int = 20            # Dense/Sparse 각각 가져올 후보 수


class SourceDocument(BaseModel):
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    retrieved_contexts: List[str]
    sources: Optional[List[SourceDocument]] = None


class IndexResponse(BaseModel):
    status: str
    files_processed: List[str]
    total_chunks: int
    bm25_indexed: int
