from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    include_sources: bool = True

class SourceDocument(BaseModel):
    content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceDocument]] = None

class EvaluationResult(BaseModel):
    id: int
    question: str
    expected_answer: str
    llm_answer: Optional[str] = None
    difficulty: str
    source_year: str
