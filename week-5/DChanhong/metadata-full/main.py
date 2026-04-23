import uvicorn
from fastapi import FastAPI

from src.core.config import settings
from src.domains.rag.router import router as rag_router

app = FastAPI(
    title="Medical Aid RAG — Metadata Full",
    description="FastAPI + LangChain RAG (Dense + BM25 + Rerank + Year Pre-filter)",
    version="1.0.0",
)

app.include_router(rag_router)


@app.get("/")
async def root():
    return {
        "message": "Medical Aid RAG API (metadata-full) is running",
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "rerank_model": settings.RERANK_MODEL,
        "reference_year": settings.REFERENCE_YEAR,
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=True)
