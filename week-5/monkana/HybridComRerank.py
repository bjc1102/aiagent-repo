import os
import time
from pathlib import Path
from typing import Any, Optional

import cohere
from dotenv import load_dotenv
from langchain_core.documents import Document

import BasicRag
import HybridCom
import HybridRerank


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "golden_dataset_step2_pilot_5.jsonl"
COMPRESSION_RERANK_TOP = 8


def cohere_rerank_compressed_docs(
    query: str,
    docs: list[Document],
    top_n: int,
) -> list[Document]:
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    candidates = [
        f"{doc.metadata.get('subject', '')}\n{doc.page_content}".strip()
        for doc in docs
    ]
    for attempt in range(3):
        try:
            results = co.rerank(
                model="rerank-v3.5",
                query=query,
                documents=candidates,
                top_n=top_n,
            )
            return [docs[r.index] for r in results.results]
        except cohere.errors.too_many_requests_error.TooManyRequestsError:
            wait = 60 * (attempt + 1)
            print(f"Rate limit 도달 — {wait}초 대기 후 재시도 ({attempt+1}/3)")
            time.sleep(wait)
    raise RuntimeError("Cohere rate limit: 3회 재시도 실패")


def retrieve_docs_hybrid_compression_rerank(
    vectorstore,
    all_documents: list[Document],
    question: str,
    source_year: str = "",
    top_n: int = COMPRESSION_RERANK_TOP,
) -> list[Document]:
    source_year = source_year or HybridRerank.infer_source_year(question)

    if "," in source_year:
        years = [y.strip() for y in source_year.split(",") if y.strip()]
        docs: list[Document] = []
        per_year_top = max(1, top_n // len(years))
        for yr in years:
            ensemble = HybridRerank.build_hybrid_rerank_retriever(
                vectorstore,
                all_documents,
                yr,
                vector_k=HybridCom.VECTOR_K,
                bm25_k=HybridCom.BM25_K,
            )
            compression_retriever = HybridCom.build_contextual_compression_retriever(
                ensemble
            )
            compressed_docs = compression_retriever.invoke(question)
            docs.extend(
                cohere_rerank_compressed_docs(question, compressed_docs, per_year_top)
            )
        return docs

    ensemble = HybridRerank.build_hybrid_rerank_retriever(
        vectorstore,
        all_documents,
        source_year,
        vector_k=HybridCom.VECTOR_K,
        bm25_k=HybridCom.BM25_K,
    )
    compression_retriever = HybridCom.build_contextual_compression_retriever(ensemble)
    compressed_docs = compression_retriever.invoke(question)
    return cohere_rerank_compressed_docs(question, compressed_docs, top_n)


def ask_question_hybrid_compression_rerank(
    vectorstore,
    all_documents: list[Document],
    question: str,
    source_year: str = "",
    top_n: int = COMPRESSION_RERANK_TOP,
    retrieved_docs: Optional[list[Document]] = None,
) -> str:
    if retrieved_docs is None:
        retrieved_docs = retrieve_docs_hybrid_compression_rerank(
            vectorstore,
            all_documents,
            question,
            source_year=source_year,
            top_n=top_n,
        )

    _, context = HybridRerank.format_retrieved_contexts(retrieved_docs)
    prompt = HybridRerank.build_prompt(question, context)
    return str(HybridRerank.get_llm().invoke(prompt).content)


def run_pipeline(
    vectorstore,
    all_documents: list[Document],
    question: str,
    source_year: str = "",
    top_n: int = COMPRESSION_RERANK_TOP,
) -> dict[str, Any]:
    retrieved_docs = retrieve_docs_hybrid_compression_rerank(
        vectorstore,
        all_documents,
        question,
        source_year=source_year,
        top_n=top_n,
    )
    retrieved_contexts, _ = HybridRerank.format_retrieved_contexts(retrieved_docs)
    response = ask_question_hybrid_compression_rerank(
        vectorstore,
        all_documents,
        question,
        source_year=source_year,
        top_n=top_n,
        retrieved_docs=retrieved_docs,
    )

    return {
        "question": question,
        "source_year": source_year or HybridRerank.infer_source_year(question),
        "retrieved_docs": retrieved_docs,
        "retrieved_contexts": retrieved_contexts,
        "response": response,
    }


def main():
    print("ChromaDB 로드 중...")
    vectorstore = BasicRag.load_vectorstore()

    print("BM25 인덱스용 문서 로드 중...")
    all_documents = HybridRerank.load_all_documents()
    print(f"  총 {len(all_documents)}개 청크 로드 완료")

    HybridCom.evaluate_golden_dataset(
        vectorstore,
        all_documents,
        DATASET_PATH,
        search_label="Hybrid (Vector+BM25) + Contextual Compression + Cohere Rerank",
        output_filename="evaluation_hybrid_com_rerank.txt",
        pipeline_name="Hybrid+Compression+Rerank",
        retrieve_fn=retrieve_docs_hybrid_compression_rerank,
        answer_fn=ask_question_hybrid_compression_rerank,
        top_n=COMPRESSION_RERANK_TOP,
        extra_settings=[
            ("Re-ranker 종류 및 모델명", "Cohere CohereRerank / rerank-v3.5"),
        ],
    )


if __name__ == "__main__":
    main()
