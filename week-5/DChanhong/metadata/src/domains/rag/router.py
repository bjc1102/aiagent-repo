from fastapi import APIRouter, HTTPException

from .schemas import QueryRequest, QueryResponse, IndexResponse
from .service import rag_service

router = APIRouter(prefix="/api/v1/rag", tags=["RAG (Metadata: Basic + Pre-filter)"])


@router.post("/index", response_model=IndexResponse)
async def run_indexing():
    """data/ 폴더의 PDF 를 읽어 벡터 DB 를 재생성합니다."""
    result = await rag_service.run_indexing()
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Metadata RAG (Basic + Year pre-filter) 로 답변을 생성합니다.
    응답에 filter_info 가 포함되어 어떤 년도로 필터링됐는지 확인 가능.
    """
    try:
        return await rag_service.get_answer(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def run_evaluation():
    """골든 데이터셋 전체에 대해 답변을 생성해서 Ragas 입력용 파일을 만듭니다."""
    result = await rag_service.run_evaluation()
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
