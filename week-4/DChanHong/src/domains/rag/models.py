from typing import List, Optional, Dict, Any

class DocumentEntity:
    """RAG 시스템에서 내부적으로 관리되는 문서 데이터 엔터티"""
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata

class RAGAnswerEntity:
    """RAG 결과 데이터를 담는 도메인 엔터티"""
    def __init__(self, answer: str, sources: Optional[List[DocumentEntity]] = None):
        self.answer = answer
        self.sources = sources
