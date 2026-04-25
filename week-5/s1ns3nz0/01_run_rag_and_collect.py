"""
Step 1 — Run Basic RAG and Advanced RAG on golden_dataset_v2.jsonl
         and collect (retrieved_contexts, response) for each question.

WHY this is split from Ragas evaluation:
  - RAG runs are expensive (LLM call per question)
  - Ragas evaluation is even more expensive (5 metrics × LLM calls per metric)
  - If we couple them and the evaluation step fails, we have to re-run the RAG
  - So we save RAG outputs to JSON, then evaluate from disk

WHAT this produces:
  - rag_outputs_basic.json
  - rag_outputs_advanced.json

  Each file: list of dicts {question, ground_truth, ground_truth_contexts,
                            response, retrieved_contexts, source_year, difficulty, id}

REUSES:
  - week-4/s1ns3nz0/data/ PDFs
  - week-4/s1ns3nz0/vectorstore_basic/ FAISS index (so we don't re-embed)
  - week-4/s1ns3nz0/{basic_rag,advanced_rag}.py logic — but we need the
    pipelines to ALSO return the retrieved chunks, not just the answer.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ============================================================
# Reuse week-4 RAG modules
# ============================================================
HERE = Path(__file__).resolve().parent
WEEK4_DIR = HERE.parent.parent / "week-4" / "s1ns3nz0"
sys.path.insert(0, str(WEEK4_DIR))

# Make week-4 paths absolute so its `BASE_DIR=os.path.dirname(__file__)` works
# even though we're running from week-5.
import basic_rag  # noqa: E402
import advanced_rag  # noqa: E402

# ============================================================
# Config
# ============================================================
GOLDEN_DATASET_V2 = HERE / "golden_dataset_v2.jsonl"
BASIC_OUT = HERE / "rag_outputs_basic.json"
ADVANCED_OUT = HERE / "rag_outputs_advanced.json"

USE_COHERE = bool(os.environ.get("COHERE_API_KEY"))
LOCAL_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # multilingual, supports Korean

# ============================================================
# Load golden dataset v2 (skip comment/blank lines)
# ============================================================
def load_dataset_v2():
    rows = []
    with open(GOLDEN_DATASET_V2, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


# ============================================================
# Run Basic RAG and capture (response, retrieved_contexts)
# ============================================================
def run_basic(samples):
    print("\n" + "=" * 70)
    print(" Running Basic RAG (vector search → Claude)")
    print("=" * 70)

    vectorstore = basic_rag.load_vectorstore()
    outputs = []
    for i, s in enumerate(samples, 1):
        qid = f"q{i:02d}"
        print(f"  [{qid}] {s['question'][:50]}…")
        answer, docs = basic_rag.retrieve_and_generate(vectorstore, s["question"])
        outputs.append({
            "id": qid,
            "question": s["question"],
            "ground_truth": s["ground_truth"],
            "ground_truth_contexts": s["ground_truth_contexts"],
            "difficulty": s["difficulty"],
            "source_year": s["source_year"],
            "response": answer,
            "retrieved_contexts": [d.page_content for d in docs],
            "retrieved_years": [d.metadata.get("source_year", "?") for d in docs],
        })
    return outputs


# ============================================================
# Run Advanced RAG and capture (response, retrieved_contexts)
#
# Path A (default if COHERE_API_KEY set): reuse week-4/advanced_rag.py
# Path B (local fallback): same hybrid retrieval but local cross-encoder
#                         re-ranker (BAAI/bge-reranker-v2-m3, multilingual).
# ============================================================
def _build_local_advanced(chunks):
    """Hybrid retriever with a local cross-encoder re-ranker."""
    from sentence_transformers import CrossEncoder
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from rank_bm25 import BM25Okapi

    embeddings = HuggingFaceEmbeddings(
        model_name=advanced_rag.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        advanced_rag.VECTORSTORE_PATH, embeddings,
        allow_dangerous_deserialization=True,
    )
    bm25_index = BM25Okapi([d.page_content.split() for d in chunks])

    print(f"[Re-ranking] Local cross-encoder: {LOCAL_RERANKER_MODEL}")
    reranker = CrossEncoder(LOCAL_RERANKER_MODEL, device="cpu")
    return vectorstore, bm25_index, chunks, reranker


def _local_hybrid_search_and_rerank(vectorstore, bm25_index, all_chunks,
                                     reranker, question, top_n=5):
    vector_docs = vectorstore.similarity_search(question, k=advanced_rag.VECTOR_K)
    bm25_scores = bm25_index.get_scores(question.split())
    top_bm25_idx = sorted(range(len(bm25_scores)),
                          key=lambda i: bm25_scores[i], reverse=True)[:advanced_rag.BM25_K]
    bm25_docs = [all_chunks[i] for i in top_bm25_idx]

    seen, merged = set(), []
    for d in vector_docs + bm25_docs:
        key = d.page_content[:200]
        if key not in seen:
            seen.add(key)
            merged.append(d)
    if not merged:
        return []

    pairs = [(question, d.page_content) for d in merged]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(merged, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]


def _local_retrieve_and_generate(vectorstore, bm25_index, all_chunks,
                                  reranker, question):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    docs = _local_hybrid_search_and_rerank(
        vectorstore, bm25_index, all_chunks, reranker, question
    )
    parts = []
    for d in docs:
        year = d.metadata.get("source_year", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"[출처: {year}년 문서, p.{page}]\n{d.page_content}")
    context = "\n\n---\n\n".join(parts)
    prompt = advanced_rag.RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    llm = ChatAnthropic(model=advanced_rag.LLM_MODEL, max_tokens=500, temperature=0)
    answer = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return answer, docs


def run_advanced(samples):
    if USE_COHERE:
        print("\n" + "=" * 70)
        print(" Running Advanced RAG (Hybrid + Cohere re-rank → Claude)")
        print("=" * 70)

        chunks = advanced_rag.load_and_chunk_pdfs()
        vectorstore, bm25_index, all_chunks, cohere_client = (
            advanced_rag.build_hybrid_retriever(chunks)
        )
        retrieve_fn = lambda q: advanced_rag.retrieve_and_generate(
            vectorstore, bm25_index, all_chunks, cohere_client, q
        )
    else:
        print("\n" + "=" * 70)
        print(" Running Advanced RAG (Hybrid + LOCAL cross-encoder re-rank → Claude)")
        print("  COHERE_API_KEY not set → using BAAI/bge-reranker-v2-m3 instead.")
        print("=" * 70)
        chunks = advanced_rag.load_and_chunk_pdfs()
        vectorstore, bm25_index, all_chunks, reranker = _build_local_advanced(chunks)
        retrieve_fn = lambda q: _local_retrieve_and_generate(
            vectorstore, bm25_index, all_chunks, reranker, q
        )

    outputs = []
    for i, s in enumerate(samples, 1):
        qid = f"q{i:02d}"
        print(f"  [{qid}] {s['question'][:50]}…")
        if USE_COHERE and i > 1 and i % 10 == 1:
            print("  …sleeping 30s for Cohere rate limit")
            time.sleep(30)
        answer, docs = retrieve_fn(s["question"])
        outputs.append({
            "id": qid,
            "question": s["question"],
            "ground_truth": s["ground_truth"],
            "ground_truth_contexts": s["ground_truth_contexts"],
            "difficulty": s["difficulty"],
            "source_year": s["source_year"],
            "response": answer,
            "retrieved_contexts": [d.page_content for d in docs],
            "retrieved_years": [d.metadata.get("source_year", "?") for d in docs],
        })
    return outputs


# ============================================================
# Main
# ============================================================
def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  saved → {path}")


def main():
    samples = load_dataset_v2()
    print(f"Loaded {len(samples)} questions from golden_dataset_v2.jsonl")

    which = sys.argv[1] if len(sys.argv) > 1 else "both"

    if which in ("basic", "both"):
        outs = run_basic(samples)
        save(BASIC_OUT, outs)

    if which in ("advanced", "both"):
        outs = run_advanced(samples)
        save(ADVANCED_OUT, outs)


if __name__ == "__main__":
    main()
