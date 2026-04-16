import os
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# --- 1. 공통 설정 ---
print("🚀 FAISS 벡터 저장소 및 LLM 로드 중...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.load_local("medical_advanced_index", embeddings, allow_dangerous_deserialization=True)

# RAG: 질문과 관련된 상위 5개 청크만 검색
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ContextLLM: 전체 문서를 한 번에 컨텍스트로 로드 (2주차 방식)
def load_full_context():
    """FAISS에 저장된 모든 청크를 년도별로 정렬하여 반환"""
    all_docs = list(vectorstore.docstore._dict.values())
    all_docs.sort(key=lambda d: (d.metadata.get("source_year", ""), d.metadata.get("page", 0)))
    return "\n\n".join([
        f"[출처: {d.metadata.get('source_year', '알수없음')}년도 / {d.metadata.get('section_title', '')}]\n{d.page_content}"
        for d in all_docs
    ])

print("📚 전체 컨텍스트 로드 중 (ContextLLM용)...")
full_context = load_full_context()
print(f"   → 총 {len(vectorstore.docstore._dict)}개 청크, {len(full_context):,}자")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- 2. 프롬프트 템플릿 ---
PROMPT_TEMPLATE = """당신은 의료급여제도 전문가입니다. 아래 컨텍스트를 바탕으로 질문에 답하세요.

지침:
1. 각 컨텍스트 상단의 [출처 년도]를 확인하세요.
2. 질문에서 요구하는 년도의 정보만 사용하여 답변하세요.
3. 두 년도의 정보가 모두 있고 수치가 다르다면, 비교해서 설명해 주세요.
4. 컨텍스트에 답변 근거가 없다면 "정보를 찾을 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def format_docs_with_metadata(docs):
    return "\n\n".join([
        f"[출처: {d.metadata.get('source_year', '알수없음')}년도]\n{d.page_content}"
        for d in docs
    ])

# --- 3. 규칙 기반 자동 채점 함수 ---
def simple_evaluator(expected, generated, target_year, retrieved_years):
    try:
        if not expected or not generated:
            return "오답", "답변 생성 실패 또는 모범 답안 누락"

        target_list = [y.strip() for y in str(target_year).split(',')]
        if target_year and "2025" in target_list and "2026" in target_list:
            pass
        elif target_year and target_year not in retrieved_years:
            return "년도 오류", f"목표 연도({target_year})가 검색된 문서 연도({retrieved_years})에 없음"

        expected_clean = str(expected).replace(",", "")
        generated_clean = str(generated).replace(",", "")

        numbers_in_expected = re.findall(r'\d+', expected_clean)

        if not numbers_in_expected:
            if any(word in expected for word in ["무료", "면제"]):
                if any(word in generated for word in ["무료", "면제"]):
                    return "정답", "핵심 키워드(무료/면제) 포함됨"
                else:
                    return "오답", "핵심 키워드 누락"
            return "수동 평가", "숫자나 명확한 키워드가 없어 판단 불가"

        all_found = all(num in generated_clean for num in numbers_in_expected)
        if all_found:
            return "정답", f"모범 답안의 수치({numbers_in_expected}) 모두 포함"
        else:
            missing = [n for n in numbers_in_expected if n not in generated_clean]
            return "오답", f"수치 누락: {missing}"

    except Exception as e:
        return "에러", f"평가 로직 오류: {str(e)}"

# --- 4. 단일 질문 평가 ---
def evaluate_single(question, expected_answer, target_year, mode):
    """
    mode: "rag" 또는 "context_llm"
    반환: (generated_answer, retrieved_years, status, reason)
    """
    if mode == "rag":
        docs = retriever.invoke(question)
        context = format_docs_with_metadata(docs)
        retrieved_years = list(set([d.metadata.get('source_year') for d in docs]))
    else:  # context_llm
        context = full_context
        retrieved_years = ["2025", "2026"]  # 전체 문서 포함

    chain = prompt | llm
    generated_answer = chain.invoke({"context": context, "question": question}).content
    status, reason = simple_evaluator(expected_answer, generated_answer, target_year, retrieved_years)
    return generated_answer, retrieved_years, status, reason

# --- 5. 통계 계산 헬퍼 ---
def calc_stats(results):
    correct    = sum(1 for r in results if r["status"] == "정답")
    year_err   = sum(1 for r in results if r["status"] == "년도 오류")
    incorrect  = sum(1 for r in results if r["status"] == "오답")
    manual     = sum(1 for r in results if r["status"] == "수동 평가")
    total      = len(results)
    rate       = correct / total * 100 if total else 0
    return {"정답": correct, "년도 오류": year_err, "오답": incorrect,
            "수동 평가": manual, "합계": total, "정답률": rate}

# --- 6. 메인 평가 실행 ---
def run_evaluation():
    dataset_path = "golden_dataset.json"
    output_path  = "result.json"

    if not os.path.exists(dataset_path):
        print(f"⚠️ {dataset_path} 파일을 찾을 수 없습니다.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    total = len(golden_dataset)
    print(f"\n🎯 총 {total}개 질문 — Basic RAG vs ContextLLM 비교 평가 시작\n")

    rag_results     = []
    context_results = []

    for idx, item in enumerate(golden_dataset):
        q_id             = item["id"]
        question         = item["question"]
        expected_answer  = item["expected_answer"]
        target_year      = item.get("source_year", "")
        difficulty       = item.get("difficulty", "unknown")

        print(f"[{idx+1}/{total}] {q_id} ({difficulty}) 진행 중...")

        # Basic RAG 평가
        rag_ans, rag_years, rag_status, rag_reason = evaluate_single(
            question, expected_answer, target_year, mode="rag"
        )

        # ContextLLM 평가
        ctx_ans, ctx_years, ctx_status, ctx_reason = evaluate_single(
            question, expected_answer, target_year, mode="context_llm"
        )

        rag_results.append({
            "id": q_id, "difficulty": difficulty, "target_year": target_year,
            "retrieved_years": rag_years, "question": question,
            "expected_answer": expected_answer,
            "generated_answer": rag_ans.strip(),
            "status": rag_status, "reason": rag_reason
        })
        context_results.append({
            "id": q_id, "difficulty": difficulty, "target_year": target_year,
            "retrieved_years": ctx_years, "question": question,
            "expected_answer": expected_answer,
            "generated_answer": ctx_ans.strip(),
            "status": ctx_status, "reason": ctx_reason
        })

        # 진행 상황 미리 보기
        match = "✅" if rag_status == ctx_status else (
            "🔵 RAG wins" if rag_status == "정답" else "🟡 CTX wins"
        )
        print(f"   RAG: {rag_status} | ContextLLM: {ctx_status}  {match}")

    # --- 7. 결과 저장 ---
    rag_stats = calc_stats(rag_results)
    ctx_stats = calc_stats(context_results)

    output = {
        "summary": {
            "basic_rag":    rag_stats,
            "context_llm":  ctx_stats
        },
        "details": {
            "basic_rag":    rag_results,
            "context_llm":  context_results
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    # --- 8. 비교 결과 출력 ---
    w = 20
    print("\n" + "=" * 52)
    print("📊 Basic RAG vs ContextLLM 비교 결과")
    print("=" * 52)
    print(f"{'항목':<{w}} {'Basic RAG':>12} {'ContextLLM':>12}")
    print("-" * 52)
    for key in ["정답", "년도 오류", "오답", "수동 평가", "합계"]:
        print(f"{'✅ '+key if key=='정답' else key:<{w}} {rag_stats[key]:>12} {ctx_stats[key]:>12}")
    print("-" * 52)
    print(f"{'🏆 정답률':<{w}} {rag_stats['정답률']:>11.1f}% {ctx_stats['정답률']:>11.1f}%")
    print("=" * 52)

    # 난이도별 비교
    difficulties = sorted(set(r["difficulty"] for r in rag_results))
    print("\n📈 난이도별 정답률 비교")
    print(f"{'난이도':<{w}} {'Basic RAG':>12} {'ContextLLM':>12}")
    print("-" * 52)
    for diff in difficulties:
        r_diff = [r for r in rag_results     if r["difficulty"] == diff]
        c_diff = [r for r in context_results if r["difficulty"] == diff]
        r_rate = sum(1 for r in r_diff if r["status"] == "정답") / len(r_diff) * 100
        c_rate = sum(1 for r in c_diff if r["status"] == "정답") / len(c_diff) * 100
        print(f"{diff:<{w}} {r_rate:>11.1f}% {c_rate:>11.1f}%")
    print("=" * 52)

    # 두 모델이 다른 결과를 낸 문항 목록
    diverged = [
        (rag_results[i], context_results[i])
        for i in range(total)
        if rag_results[i]["status"] != context_results[i]["status"]
    ]
    if diverged:
        print(f"\n🔍 두 모델 결과가 다른 문항 ({len(diverged)}개)")
        for r, c in diverged:
            print(f"  {r['id']} ({r['difficulty']}) | RAG: {r['status']} / ContextLLM: {c['status']}")

    print(f"\n📁 상세 결과 저장: '{output_path}'")

if __name__ == "__main__":
    run_evaluation()
