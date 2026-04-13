import uvicorn
from fastapi import FastAPI
from src.core.config import settings
from src.domains.rag.router import router as rag_router

app = FastAPI(
    title="Medical Aid RAG Service",
    description="FastAPI + LangChain RAG system for Medical Aid",
    version="1.0.0"
)

# API 라우터 등록
app.include_router(rag_router)

@app.get("/")
async def root():
    return {
        "message": "Medical Aid RAG API is running",
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
