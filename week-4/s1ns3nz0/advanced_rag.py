"""
Step 2: Advanced RAG — Hybrid Search (Vector + BM25) + Re-ranking (Cohere)

What's different from Basic RAG:
  Basic:    Question → Vector Search → Top-5 → Claude
  Advanced: Question → [Vector + BM25] → Merge 20 candidates → Re-rank → Top-5 → Claude

Why:
  - BM25 catches exact keyword matches that vector search misses (e.g., "추나요법", "KTAS")
  - Re-ranker re-scores all candidates with a powerful cross-encoder model
    so the most relevant chunks float to the top
"""
import json
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from rank_bm25 import BM25Okapi
import cohere
import time

# ============================================================
# Config
# ============================================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore_basic")
GOLDEN_DATASET_PATH = os.path.join(BASE_DIR, "golden_dataset.jsonl")
RESULTS_PATH = os.path.join(BASE_DIR, "advanced_rag_results.json")

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
LLM_MODEL = "claude-sonnet-4-20250514"
RERANK_MODEL = "rerank-v3.5"

PDF_FILES = {
    "2025": os.path.join(DATA_DIR, "2025 알기 쉬운 의료급여제도.pdf"),
    "2026": os.path.join(DATA_DIR, "2026 알기 쉬운 의료급여제도.pdf"),
}

# --- Hybrid Search Settings ---
VECTOR_K = 10          # candidates from vector search
BM25_K = 10            # candidates from BM25 search
ENSEMBLE_WEIGHTS = [0.5, 0.5]  # equal weight: vector and BM25
RERANK_TOP_N = 5       # final number of chunks after re-ranking

# API keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    env_path = os.path.join(BASE_DIR, "..", "..", "1week", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    ANTHROPIC_API_KEY = line.strip().split("=", 1)[1]
                    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY


# ============================================================
# Part 1: Load chunks (needed for BM25 — it works in-memory, not from disk)
# ============================================================
def load_and_chunk_pdfs(chunk_size=500, chunk_overlap=100):
    """Same as basic_rag.py — load both PDFs, tag with source_year, chunk."""
    all_chunks = []
    for year, pdf_path in PDF_FILES.items():
        print(f"[로딩] {year}년 PDF")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        for page in pages:
            page.metadata["source_year"] = year
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(pages)
        print(f"  → {len(chunks)}개 청크")
        all_chunks.extend(chunks)
    print(f"[전체] 총 {len(all_chunks)}개 청크")
    return all_chunks


# ============================================================
# Part 2: Build Hybrid Retriever (Vector + BM25) + Re-ranker
# ============================================================
def build_hybrid_retriever(chunks):
    """
    Build the Advanced RAG retriever in 3 layers:

    Layer 1 — Vector Retriever (semantic search via FAISS)
      Finds chunks with similar MEANING to the question.
      Good at: paraphrases, synonyms, related concepts
      Bad at:  exact keyword matching ("KTAS", "추나요법")

    Layer 2 — BM25 Retriever (keyword search, in-memory)
      Finds chunks containing the same WORDS as the question.
      Good at: exact terms, numbers, domain-specific jargon
      Bad at:  paraphrases, different wording for same concept

    Layer 3 — Re-rank (Cohere API)
      Takes all candidates from both retrievers, re-scores with a cross-encoder,
      and returns only the top_n most relevant.
    """

    # --- Layer 1: Vector store (reuse existing FAISS) ---
    print(f"\n[벡터 검색] FAISS 로드: {VECTORSTORE_PATH}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

    # --- Layer 2: BM25 index (built in-memory from chunks) ---
    # BM25 tokenizes each chunk into words and builds an inverted index
    print(f"[BM25 검색] {len(chunks)}개 청크로 BM25 인덱스 구축")
    tokenized_chunks = [doc.page_content.split() for doc in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)

    # --- Layer 3: Cohere re-ranker client ---
    print(f"[Re-ranking] Cohere {RERANK_MODEL}, top_n={RERANK_TOP_N}")
    cohere_client = cohere.Client(os.environ.get("COHERE_API_KEY"))

    return vectorstore, bm25_index, chunks, cohere_client


def hybrid_search_and_rerank(vectorstore, bm25_index, all_chunks, cohere_client, question):
    """
    The full Advanced RAG retrieval pipeline:

    Step 1: Vector search → top VECTOR_K candidates
    Step 2: BM25 search   → top BM25_K candidates
    Step 3: Merge both lists, remove duplicates
    Step 4: Re-rank merged candidates with Cohere → top RERANK_TOP_N
    """

    # Step 1: Vector search (semantic)
    vector_docs = vectorstore.similarity_search(question, k=VECTOR_K)

    # Step 2: BM25 search (keyword)
    tokenized_query = question.split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    # Get indices of top BM25_K scores
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_K]
    bm25_docs = [all_chunks[i] for i in top_bm25_indices]

    # Step 3: Merge and deduplicate
    seen_contents = set()
    merged_docs = []
    for doc in vector_docs + bm25_docs:
        content_key = doc.page_content[:200]  # use first 200 chars as dedup key
        if content_key not in seen_contents:
            seen_contents.add(content_key)
            merged_docs.append(doc)

    # Step 4: Re-rank with Cohere (with rate limit handling for free tier: 10 calls/min)
    if not merged_docs:
        return []

    for attempt in range(3):
        try:
            rerank_response = cohere_client.rerank(
        model=RERANK_MODEL,
        query=question,
        documents=[doc.page_content for doc in merged_docs],
            top_n=RERANK_TOP_N,
            )
            break
        except cohere.errors.too_many_requests_error.TooManyRequestsError:
            wait = 15 * (attempt + 1)
            print(f"    ⏳ Rate limited, waiting {wait}s...")
            time.sleep(wait)

    # Return the top re-ranked documents (with original metadata preserved)
    reranked_docs = []
    for result in rerank_response.results:
        doc = merged_docs[result.index]
        reranked_docs.append(doc)

    return reranked_docs


# ============================================================
# Part 3: Generation — Same prompt as Basic RAG
# ============================================================
RAG_PROMPT_TEMPLATE = """아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 출처 년도가 표시되어 있습니다. 질문이 특정 년도를 묻는 경우 해당 년도의 정보만 사용하세요.
컨텍스트에 없는 내용은 "정보를 찾을 수 없습니다"라고 답하세요.
답변은 간결하게 핵심만 답하세요.

컨텍스트:
{context}

질문: {question}

답변:"""


def retrieve_and_generate(vectorstore, bm25_index, all_chunks, cohere_client, question):
    """Run the full Advanced RAG pipeline for one question."""
    # RETRIEVE: Hybrid search + re-ranking
    docs = hybrid_search_and_rerank(vectorstore, bm25_index, all_chunks, cohere_client, question)

    # BUILD CONTEXT
    context_parts = []
    for i, doc in enumerate(docs, 1):
        year = doc.metadata.get("source_year", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[출처: {year}년 문서, p.{page}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # GENERATE
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    llm = ChatAnthropic(model=LLM_MODEL, max_tokens=500, temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()

    return answer, docs


# ============================================================
# Part 4: Evaluation
# ============================================================
def load_golden_dataset():
    questions = []
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            questions.append(json.loads(line))
    return questions


def check_answer(expected, generated):
    expected_lower = expected.lower().replace(" ", "")
    generated_lower = generated.lower().replace(" ", "")
    key_parts = expected.replace("→", ",").replace("~", ",").replace("에서", ",").split(",")
    key_parts = [p.strip() for p in key_parts if len(p.strip()) >= 2]
    if not key_parts:
        return expected_lower in generated_lower
    matched = sum(1 for p in key_parts if p.replace(" ", "").lower() in generated_lower)
    return matched >= len(key_parts) * 0.5


def check_year_retrieval(docs, source_year):
    retrieved_years = {doc.metadata.get("source_year", "?") for doc in docs}
    if "+" in source_year:
        needed = set(source_year.split("+"))
        return needed.issubset(retrieved_years)
    else:
        return source_year in retrieved_years


def evaluate(vectorstore, bm25_index, all_chunks, cohere_client):
    questions = load_golden_dataset()
    results = []
    correct_count = 0
    year_correct_count = 0

    print(f"\n{'='*70}")
    print(f"Advanced RAG 평가 시작 ({len(questions)}문항)")
    print(f"{'='*70}")

    for q in questions:
        qid = f"q{len(results)+1:02d}"
        print(f"\n[{qid}] ({q['difficulty']}) {q['question']}")

        answer, docs = retrieve_and_generate(vectorstore, bm25_index, all_chunks, cohere_client, q["question"])

        is_correct = check_answer(q["expected_answer"], answer)
        is_year_correct = check_year_retrieval(docs, q["source_year"])

        if is_correct:
            correct_count += 1
        if is_year_correct:
            year_correct_count += 1

        result = {
            "id": qid,
            "difficulty": q["difficulty"],
            "source_year": q["source_year"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "generated_answer": answer,
            "is_correct": is_correct,
            "is_year_correct": is_year_correct,
            "retrieved_years": [doc.metadata.get("source_year", "?") for doc in docs],
            "retrieved_chunks": [doc.page_content[:150] for doc in docs],
        }
        results.append(result)

        status = "✅ 정답" if is_correct else "❌ 오답"
        year_status = "✅" if is_year_correct else "❌"
        print(f"  기대: {q['expected_answer']}")
        print(f"  생성: {answer}")
        print(f"  판정: {status} | 년도검색: {year_status}")

    total = len(questions)
    print(f"\n{'='*70}")
    print(f"[Advanced RAG 결과]")
    print(f"  정답률:       {correct_count}/{total} ({correct_count/total*100:.1f}%)")
    print(f"  년도 정확도:  {year_correct_count}/{total} ({year_correct_count/total*100:.1f}%)")
    print(f"{'='*70}")

    output = {
        "pipeline": "Advanced RAG (Hybrid + Re-ranking)",
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "vector_store": "FAISS",
            "bm25": "rank_bm25 (in-memory)",
            "reranker": f"Cohere {RERANK_MODEL}",
            "chunk_size": 500,
            "chunk_overlap": 100,
            "vector_k": VECTOR_K,
            "bm25_k": BM25_K,
            "ensemble_weights": ENSEMBLE_WEIGHTS,
            "rerank_top_n": RERANK_TOP_N,
        },
        "summary": {
            "total": total,
            "correct": correct_count,
            "accuracy": f"{correct_count/total*100:.1f}%",
            "year_correct": year_correct_count,
            "year_accuracy": f"{year_correct_count/total*100:.1f}%",
        },
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[결과 저장] {RESULTS_PATH}")

    return output


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Step 2: Advanced RAG Pipeline")
    print(f"  임베딩:      {EMBEDDING_MODEL}")
    print(f"  LLM:        {LLM_MODEL}")
    print(f"  Re-ranker:  Cohere {RERANK_MODEL}")
    print(f"  Hybrid:     Vector({ENSEMBLE_WEIGHTS[0]}) + BM25({ENSEMBLE_WEIGHTS[1]})")
    print("=" * 70)

    # Load chunks (needed for BM25 — it can't load from FAISS)
    chunks = load_and_chunk_pdfs()

    # Build the 3-layer retriever: Vector + BM25 + Re-rank
    vectorstore, bm25_index, all_chunks, cohere_client = build_hybrid_retriever(chunks)

    # Evaluate
    evaluate(vectorstore, bm25_index, all_chunks, cohere_client)
