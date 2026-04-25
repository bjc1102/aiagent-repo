from typing import List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    include_sources: bool = True
    k: int = 10


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
