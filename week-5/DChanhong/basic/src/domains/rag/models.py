from typing import List, Optional, Dict, Any


class DocumentEntity:
    """내부 도메인에서 다루는 문서 엔터티"""

    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata


class RAGAnswerEntity:
    """RAG 응답 도메인 엔터티"""

    def __init__(self, answer: str, sources: Optional[List[DocumentEntity]] = None):
        self.answer = answer
        self.sources = sources
