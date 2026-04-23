from fastapi import APIRouter, HTTPException

from .schemas import QueryRequest, QueryResponse, IndexResponse
from .service import rag_service

router = APIRouter(prefix="/api/v1/rag", tags=["RAG (Rerank)"])


@router.post("/index", response_model=IndexResponse)
async def run_indexing():
    """data/ 폴더의 PDF 를 읽어 벡터 DB + BM25 인덱스를 재생성합니다."""
    result = await rag_service.run_indexing()
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Rerank RAG (Dense + BM25 후보 → Cohere Rerank) 로 답변을 생성합니다."""
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
