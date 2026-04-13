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

@router.post("/index")
async def run_indexing():
    """
    data 폴더의 PDF 파일들을 읽어서 ChromaDB에 인덱싱합니다.
    (주의: 데이터가 중복으로 추가될 수 있으므로 필요 시에만 호출하세요.)
    """
    try:
        result = await rag_service.run_indexing()
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
