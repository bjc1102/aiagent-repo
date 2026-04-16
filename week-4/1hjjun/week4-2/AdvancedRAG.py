import os
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- 경로 설정 ---
# FAISS 인덱스와 데이터셋은 week4-1 폴더에서 생성된 것을 참조
# week4-2 폴더에서 indexing.py를 다시 실행해 인덱스를 생성해도 됩니다.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "../week4-1/medical_advanced_index")
DATASET_PATH = os.path.join(SCRIPT_DIR, "../week4-1/golden_dataset.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "advanced_result.json")

# --- 검색 파라미터 ---
VECTOR_K = 10      # 벡터 검색 후보 수
BM25_K = 10        # BM25 검색 후보 수
RERANK_TOP_N = 3   # Re-ranking 후 최종 선택 수

# =============================================================================
# 1. 기본 컴포넌트 초기화
# =============================================================================
print("🚀 Advanced RAG 시스템 초기화 중...")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
)

# 전체 문서 추출 (BM25 인덱스 구성용)
all_docs = list(vectorstore.docstore._dict.values())
all_docs.sort(key=lambda d: (d.metadata.get("source_year", ""), d.metadata.get("page", 0)))
print(f"   → 총 {len(all_docs)}개 청크 로드 완료")

# =============================================================================
# 2. BM25 인덱스 구성 (전체 문서 대상)
# =============================================================================
print("📚 BM25 인덱스 구성 중...")
bm25_retriever_full = BM25Retriever.from_documents(all_docs)
bm25_retriever_full.k = BM25_K

# =============================================================================
# 3. 기본 Hybrid Retriever (메타데이터 필터링 없음)
# =============================================================================
vector_retriever_full = vectorstore.as_retriever(search_kwargs={"k": VECTOR_K})

ensemble_retriever_full = EnsembleRetriever(
    retrievers=[vector_retriever_full, bm25_retriever_full],
    weights=[0.5, 0.5],
)

# =============================================================================
# 4. Cohere Re-ranker 설정
# =============================================================================
print("🎯 Cohere Re-ranker 설정 중...")
reranker = CohereRerank(model="rerank-v3.5", top_n=RERANK_TOP_N)

compression_retriever_full = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever_full,
)

# =============================================================================
# 5. LLM 및 프롬프트
# =============================================================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

rag_prompt = ChatPromptTemplate.from_template("""당신은 의료급여제도 전문가입니다. 아래 컨텍스트를 바탕으로 질문에 답하세요.

지침:
1. 각 컨텍스트 상단의 [출처 년도]를 확인하세요.
2. 질문에서 요구하는 년도의 정보만 사용하여 답변하세요.
3. 두 년도의 정보가 모두 있고 수치가 다르다면, 비교해서 설명해 주세요.
4. 컨텍스트에 답변 근거가 없다면 "정보를 찾을 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:""")


def format_docs(docs):
    return "\n\n".join([
        f"[출처: {d.metadata.get('source_year', '알수없음')}년도 / {d.metadata.get('section_title', '')}]\n{d.page_content}"
        for d in docs
    ])


# =============================================================================
# 6. 년도 추출 (메타데이터 필터링용)
# =============================================================================
def extract_year(question: str):
    """질문에서 '2025' 또는 '2026' 추출. 없으면 None 반환."""
    match = re.search(r"(2025|2026)", question)
    return match.group(1) if match else None


# =============================================================================
# 7. 년도별 필터 Retriever 생성 (2-2. 메타데이터 필터링)
# =============================================================================
def build_filtered_retriever(year: str) -> ContextualCompressionRetriever:
    """
    특정 년도의 문서만 대상으로 Hybrid + Re-rank Retriever를 구성합니다.
    - 벡터 검색: FAISS filter 파라미터로 source_year 필터링
    - BM25 검색: 해당 년도 청크만 로드하여 별도 인덱스 구성
    """
    # 벡터 검색 (메타데이터 필터)
    filtered_vector = vectorstore.as_retriever(
        search_kwargs={"k": VECTOR_K, "filter": {"source_year": year}}
    )

    # BM25 (해당 년도 문서만)
    year_docs = [d for d in all_docs if d.metadata.get("source_year") == year]
    if not year_docs:
        # 해당 년도 문서가 없으면 전체로 폴백
        filtered_bm25 = bm25_retriever_full
    else:
        filtered_bm25 = BM25Retriever.from_documents(year_docs)
        filtered_bm25.k = BM25_K

    ensemble = EnsembleRetriever(
        retrievers=[filtered_vector, filtered_bm25],
        weights=[0.5, 0.5],
    )
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )


# 두 년도 필터 Retriever 사전 생성
filtered_retrievers = {
    "2025": build_filtered_retriever("2025"),
    "2026": build_filtered_retriever("2026"),
}
print("✅ 초기화 완료 (기본 Hybrid + 년도별 필터 Retriever 준비됨)\n")


# =============================================================================
# 8. 평가 실행
# =============================================================================
def run_advanced_rag_evaluation():
    if not os.path.exists(DATASET_PATH):
        print(f"⚠️ {DATASET_PATH} 파일을 찾을 수 없습니다.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    total = len(golden_dataset)
    print(f"🎯 총 {total}개 질문 — Advanced RAG 평가 시작")
    print(f"   검색: 벡터 Top-{VECTOR_K} + BM25 Top-{BM25_K} → Cohere Rerank Top-{RERANK_TOP_N}\n")

    chain = rag_prompt | llm
    results = []

    for idx, item in enumerate(golden_dataset):
        q_id = item["id"]
        question = item["question"]
        expected_answer = item["expected_answer"]
        difficulty = item.get("difficulty", "unknown")
        source_year = item.get("source_year", "")

        print(f"[{idx+1}/{total}] {q_id} ({difficulty}) 진행 중...")

        # 2-2. 메타데이터 필터링: 질문에서 단일 년도 감지 시 필터 Retriever 사용
        detected_year = extract_year(question)
        # 두 년도를 모두 포함하는 질문(예: "2025년과 2026년 비교")은 필터링 제외
        if detected_year and "2025" in question and "2026" in question:
            detected_year = None

        if detected_year and detected_year in filtered_retrievers:
            retriever = filtered_retrievers[detected_year]
            filter_used = f"year={detected_year}"
        else:
            retriever = compression_retriever_full
            filter_used = "none"

        # 2-1 & 2-3: Hybrid Search + Re-ranking
        docs = retriever.invoke(question)
        retrieved_years = list(set([d.metadata.get("source_year") for d in docs]))
        context = format_docs(docs)

        # 답변 생성
        generated_answer = chain.invoke(
            {"context": context, "question": question}
        ).content.strip()

        results.append({
            "id": q_id,
            "difficulty": difficulty,
            "source_year": source_year,
            "retrieved_years": retrieved_years,
            "metadata_filter": filter_used,
            "num_docs_used": len(docs),
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
        })

        print(f"   → 필터: {filter_used} | 검색 년도: {retrieved_years} | 문서 {len(docs)}개")

    # 결과 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 완료! 결과가 '{OUTPUT_PATH}'에 저장되었습니다.")
    print(f"   총 {total}개 질문 처리 완료.")


if __name__ == "__main__":
    run_advanced_rag_evaluation()
