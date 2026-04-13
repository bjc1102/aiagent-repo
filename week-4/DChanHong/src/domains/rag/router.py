from fastapi import APIRouter, HTTPException
from .schemas import QueryRequest, QueryResponse
from .service import rag_service

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])

@router.post("/query", response_model=QueryResponse)
async def query_medical_aid(request: QueryRequest):
    """
    의료급여제도에 대해 질문하고 RAG 기반 답변을 받습니다.
    """
    try:
        response = await rag_service.get_answer(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
