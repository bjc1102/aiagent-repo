import uvicorn
from fastapi import FastAPI

from src.core.config import settings
from src.domains.rag.router import router as rag_router

app = FastAPI(
    title="Medical Aid RAG — Metadata (Basic + Pre-filter)",
    description="FastAPI + LangChain RAG with year-based pre-retrieval filtering (Dense only)",
    version="1.0.0",
)

app.include_router(rag_router)


@app.get("/")
async def root():
    return {
        "message": "Medical Aid RAG API (metadata: basic + pre-filter) is running",
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "reference_year": settings.REFERENCE_YEAR,
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
