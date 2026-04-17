"""
Step 1: Basic RAG — Indexing + Generation + Evaluation

Pipeline:
  [Indexing]    2025 PDF + 2026 PDF → chunk with source_year metadata → embed → FAISS
  [Generation]  Question → vector search (Top-K) → context + prompt → Claude → answer
  [Evaluation]  Run all golden dataset questions → measure accuracy + year correctness
"""
import json
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# ============================================================
# Config
# ============================================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore_basic")
GOLDEN_DATASET_PATH = os.path.join(BASE_DIR, "golden_dataset.jsonl")
RESULTS_PATH = os.path.join(BASE_DIR, "basic_rag_results.json")

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
LLM_MODEL = "claude-sonnet-4-20250514"

PDF_FILES = {
    "2025": os.path.join(DATA_DIR, "2025 알기 쉬운 의료급여제도.pdf"),
    "2026": os.path.join(DATA_DIR, "2026 알기 쉬운 의료급여제도.pdf"),
}

# Anthropic API key — loaded from environment or 1week/.env
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
# Part 1: Indexing — PDF → Chunks with source_year → FAISS
# ============================================================
def load_and_chunk_pdfs(chunk_size=500, chunk_overlap=100):
    """Load both PDFs, tag each page with source_year, then chunk."""
    all_chunks = []

    for year, pdf_path in PDF_FILES.items():
        print(f"\n[로딩] {year}년 PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"  → {len(pages)}페이지 로딩 완료")

        # TAG: Add source_year to every page's metadata
        # This is critical — without it, we can't tell which year a chunk came from
        for page in pages:
            page.metadata["source_year"] = year

        # CHUNK: Split pages into smaller pieces for precise retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(pages)
        print(f"  → {len(chunks)}개 청크 생성 (chunk_size={chunk_size}, overlap={chunk_overlap})")
        all_chunks.extend(chunks)

    print(f"\n[전체] 총 {len(all_chunks)}개 청크 (2025 + 2026)")
    return all_chunks


def build_vectorstore(chunks):
    """Embed all chunks and save to FAISS vector store."""
    print(f"\n[임베딩] 모델: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"[저장] FAISS 벡터 저장소 → {VECTORSTORE_PATH}")
    return vectorstore


def load_vectorstore():
    """Load previously saved FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)


# ============================================================
# Part 2: Generation — Question → Retrieve → LLM → Answer
# ============================================================
RAG_PROMPT_TEMPLATE = """아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 출처 년도가 표시되어 있습니다. 질문이 특정 년도를 묻는 경우 해당 년도의 정보만 사용하세요.
컨텍스트에 없는 내용은 "정보를 찾을 수 없습니다"라고 답하세요.
답변은 간결하게 핵심만 답하세요.

컨텍스트:
{context}

질문: {question}

답변:"""


def retrieve_and_generate(vectorstore, question, top_k=5):
    """
    Core RAG function:
    1. Vector search for top_k most similar chunks
    2. Build context string with year labels
    3. Send prompt to Claude
    4. Return answer + retrieved docs (for evaluation)
    """
    # RETRIEVE: Find top_k chunks most similar to the question
    docs = vectorstore.similarity_search(question, k=top_k)

    # BUILD CONTEXT: Include source_year so the LLM knows which year each chunk is from
    context_parts = []
    for i, doc in enumerate(docs, 1):
        year = doc.metadata.get("source_year", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[출처: {year}년 문서, p.{page}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # GENERATE: Send to Claude
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    llm = ChatAnthropic(model=LLM_MODEL, max_tokens=500, temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()

    return answer, docs


# ============================================================
# Part 3: Evaluation — Run golden dataset, measure accuracy
# ============================================================
def load_golden_dataset():
    """Load golden dataset, skipping comment lines."""
    questions = []
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            questions.append(json.loads(line))
    return questions


def check_answer(expected, generated):
    """Simple check: does the generated answer contain the key value from expected?"""
    # Normalize for comparison
    expected_lower = expected.lower().replace(" ", "")
    generated_lower = generated.lower().replace(" ", "")

    # Check if key parts of expected answer appear in generated answer
    # Split expected by common delimiters to get key values
    key_parts = expected.replace("→", ",").replace("~", ",").replace("에서", ",").split(",")
    key_parts = [p.strip() for p in key_parts if len(p.strip()) >= 2]

    if not key_parts:
        return expected_lower in generated_lower

    matched = sum(1 for p in key_parts if p.replace(" ", "").lower() in generated_lower)
    return matched >= len(key_parts) * 0.5


def check_year_retrieval(docs, source_year):
    """Check if retrieved docs include chunks from the correct year(s)."""
    retrieved_years = {doc.metadata.get("source_year", "?") for doc in docs}

    if "+" in source_year:
        # Cross-year question: need both years
        needed = set(source_year.split("+"))
        return needed.issubset(retrieved_years)
    else:
        return source_year in retrieved_years


def evaluate(vectorstore):
    """Run all golden dataset questions through the RAG pipeline and measure accuracy."""
    questions = load_golden_dataset()
    results = []
    correct_count = 0
    year_correct_count = 0

    print(f"\n{'='*70}")
    print(f"Basic RAG 평가 시작 ({len(questions)}문항)")
    print(f"{'='*70}")

    for q in questions:
        qid = f"q{len(results)+1:02d}"
        print(f"\n[{qid}] ({q['difficulty']}) {q['question']}")

        answer, docs = retrieve_and_generate(vectorstore, q["question"])

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

    # Summary
    total = len(questions)
    print(f"\n{'='*70}")
    print(f"[Basic RAG 결과]")
    print(f"  정답률:       {correct_count}/{total} ({correct_count/total*100:.1f}%)")
    print(f"  년도 정확도:  {year_correct_count}/{total} ({year_correct_count/total*100:.1f}%)")
    print(f"{'='*70}")

    # Save results
    output = {
        "pipeline": "Basic RAG",
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "vector_store": "FAISS",
            "chunk_size": 500,
            "chunk_overlap": 100,
            "top_k": 5,
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
    import sys

    print("=" * 70)
    print("Step 1: Basic RAG Pipeline")
    print(f"  임베딩:  {EMBEDDING_MODEL}")
    print(f"  LLM:    {LLM_MODEL}")
    print("=" * 70)

    # Check if vectorstore already exists
    if os.path.exists(VECTORSTORE_PATH) and "--rebuild" not in sys.argv:
        print(f"\n[벡터 저장소 로드] {VECTORSTORE_PATH}")
        vectorstore = load_vectorstore()
    else:
        # Part 1: Index both PDFs
        chunks = load_and_chunk_pdfs()
        vectorstore = build_vectorstore(chunks)

    # Part 2+3: Evaluate with golden dataset
    evaluate(vectorstore)
