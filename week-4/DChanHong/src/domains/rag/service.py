from src.core.config import settings
from .schemas import QueryRequest, QueryResponse, SourceDocument

class RAGService:
    def __init__(self):
        self.model_name = settings.LLM_MODEL
        # TODO: LangChain 초기화 코드
        pass

    async def get_answer(self, request: QueryRequest) -> QueryResponse:
        # 비즈니스 로직 처리
        mock_answer = f"[{self.model_name}] '{request.question}'에 대한 의료급여제도 답변입니다."
        mock_sources = [
            SourceDocument(content="의료급여 1종 수급권자 대상자...", metadata={"source": "2024_의료급여제도.pdf", "page": 10})
        ] if request.include_sources else None
        
        return QueryResponse(
            answer=mock_answer,
            sources=mock_sources
        )

# 싱글톤 인스턴스
rag_service = RAGService()
