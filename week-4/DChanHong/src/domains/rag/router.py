from fastapi import APIRouter, HTTPException
from .schemas import QueryRequest, QueryResponse
from .service import rag_service

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])

@router.post("/query", response_model=QueryResponse)
async def query_medical_aid(request: QueryRequest):
    """
    기본 벡터 검색 RAG 기반 답변을 받습니다.
    """
    try:
        response = await rag_service.get_answer(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hybrid-query", response_model=QueryResponse)
async def query_medical_aid_hybrid(request: QueryRequest):
    """
    하이브리드 검색(Vector + BM25) RAG 기반 답변을 받습니다.
    """
    try:
        response = await rag_service.get_hybrid_answer(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/basic")
async def run_basic_evaluation():
    """
    Basic RAG 엔진으로 골든 데이터셋 전체를 평가합니다.
    결과는 data/basic/{index} 폴더에 저장됩니다.
    """
    try:
        result = await rag_service.run_evaluation("basic")
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hybrid")
async def run_hybrid_evaluation():
    """
    Hybrid RAG 엔진으로 골든 데이터셋 전체를 평가합니다.
    결과는 data/hybrid/{index} 폴더에 저장됩니다.
    """
    try:
        result = await rag_service.run_evaluation("hybrid")
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate/{endpoint_name}")
async def run_evaluation_by_name(endpoint_name: str):
    """
    지정한 이름(basic, hybrid 등)으로 골든 데이터셋 평가를 실행합니다.
    """
    try:
        result = await rag_service.run_evaluation(endpoint_name)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/index")
async def run_indexing():
    """
    PDF 파일들을 읽어서 벡터 DB 및 BM25 인덱스를 생성합니다.
    """
    try:
        result = await rag_service.run_indexing()
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
