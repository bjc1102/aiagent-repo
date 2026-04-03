"""
RAG Indexing Pipeline: PDF → Chunking → Embedding → Vector Store → Search Quality Check
Uses HuggingFace sentence-transformers for local embedding (no API key required).
"""
import json
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "2024 알기 쉬운 의료급여제도.pdf")
GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.jsonl")
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "vectorstore")

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# ============================================================
# Step 2-1: PDF 로딩 및 청킹
# ============================================================
def load_and_chunk(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 100):
    """PDF를 로딩하고 청킹하여 Document 리스트를 반환한다."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"[로딩] 총 {len(pages)}페이지 로딩 완료")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"[청킹] chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"[청킹] 총 {len(chunks)}개 청크 생성")
    return chunks


# ============================================================
# Step 2-2: 임베딩 및 벡터 저장소
# ============================================================
def get_embeddings():
    """HuggingFace 다국어 임베딩 모델을 반환한다."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks, save_path: str):
    """청크를 임베딩하여 FAISS 벡터 저장소에 저장한다."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"[벡터 저장소] FAISS에 {len(chunks)}개 청크 저장 완료 → {save_path}")
    return vectorstore


def load_vectorstore(save_path: str):
    """저장된 FAISS 벡터 저장소를 로드한다."""
    embeddings = get_embeddings()
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)


# ============================================================
# Step 3: 검색 품질 확인
# ============================================================
def evaluate_search(vectorstore, golden_path: str, top_k: int = 3):
    """Golden Dataset 질문에 대해 검색 품질을 평가한다."""
    with open(golden_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    results = []
    success_count = 0

    for q in questions:
        docs = vectorstore.similarity_search(q["question"], k=top_k)
        retrieved_texts = [doc.page_content for doc in docs]
        combined = "\n".join(retrieved_texts)

        # evidence_text의 핵심 키워드가 검색 결과에 포함되는지 확인
        evidence = q["evidence_text"]
        keywords = [kw.strip() for kw in evidence.replace("→", ",").replace("=", ",").split(",") if len(kw.strip()) >= 2]
        matched_keywords = [kw for kw in keywords if kw in combined]
        is_success = len(matched_keywords) >= len(keywords) * 0.5

        result = {
            "id": q["id"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "search_result": "성공" if is_success else "실패",
            "matched_keywords": matched_keywords,
            "total_keywords": keywords,
            "retrieved_chunks_summary": [text[:150] + "..." for text in retrieved_texts],
        }
        results.append(result)
        if is_success:
            success_count += 1

        print(f"\n{'='*60}")
        print(f"[{q['id']}] ({q['difficulty']}) {q['question']}")
        print(f"  기대 답변: {q['expected_answer']}")
        print(f"  검색 결과: {'✅ 성공' if is_success else '❌ 실패'}")
        print(f"  매칭 키워드: {matched_keywords} / {keywords}")
        for i, text in enumerate(retrieved_texts):
            print(f"  --- 검색 청크 {i+1} ---")
            print(f"  {text[:200]}")

    print(f"\n{'='*60}")
    print(f"[검색 성공률] {success_count}/{len(questions)}")
    return results, success_count, len(questions)


# ============================================================
# 표 데이터 포함 청크 샘플 출력
# ============================================================
def print_table_chunk_samples(chunks, num_samples=3):
    """표 데이터가 포함된 청크 샘플을 출력한다."""
    table_keywords = ["본인부담률", "본인부담금", "1종", "2종", "입원", "외래", "무료", "무 료"]
    table_chunks = [c for c in chunks if any(kw in c.page_content for kw in table_keywords)]
    print(f"\n[표 데이터 포함 청크] {len(table_chunks)}개 발견 (전체 {len(chunks)}개 중)")
    for i, chunk in enumerate(table_chunks[:num_samples]):
        print(f"\n--- 샘플 {i+1} (page {chunk.metadata.get('page', '?')}) ---")
        print(chunk.page_content[:300])


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Indexing Pipeline 시작")
    print(f"임베딩 모델: {EMBEDDING_MODEL}")
    print("=" * 60)

    # Step 2-1: PDF 로딩 및 청킹
    chunks = load_and_chunk(PDF_PATH, chunk_size=500, chunk_overlap=100)
    print_table_chunk_samples(chunks)

    # Step 2-2: 임베딩 및 벡터 저장소
    vectorstore = build_vectorstore(chunks, VECTORSTORE_PATH)

    # Step 3: 검색 품질 확인
    results, success, total = evaluate_search(vectorstore, GOLDEN_DATASET_PATH, top_k=3)

    # 결과를 JSON으로 저장
    output_path = os.path.join(os.path.dirname(__file__), "search_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "embedding_model": EMBEDDING_MODEL,
                "vector_store": "FAISS",
                "chunk_size": 500,
                "chunk_overlap": 100,
                "top_k": 3,
            },
            "search_success_rate": f"{success}/{total}",
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[결과 저장] {output_path}")
