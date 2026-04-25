import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- 경로 설정 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "../medical_advanced_index")
DATASET_PATH = os.path.join(SCRIPT_DIR, "../golden_dataset.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "RAGresult.json")

# --- 1. FAISS 벡터 저장소 및 LLM 로드 ---
print("🚀 FAISS 벡터 저장소 및 LLM 로드 중...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

rag_prompt = ChatPromptTemplate.from_template("""당신은 의료급여제도 전문가입니다. 아래 컨텍스트를 바탕으로 질문에 답하세요.

지침:
0. 답변은 참고자료를 그대로 복사붙여넣기 하지말고 질문에 맞게 참고자료의 내용을 이해하여 재구성해서 작성하세요. 간단하게 
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
        f"[출처: {d.metadata.get('source_year', '알수없음')}년도]\n{d.page_content}"
        for d in docs
    ])


# --- 2. 평가 실행 ---
def run_rag_evaluation():
    dataset_path = DATASET_PATH
    output_path  = OUTPUT_PATH

    if not os.path.exists(dataset_path):
        print(f"⚠️ {dataset_path} 파일을 찾을 수 없습니다.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    results = []
    total = len(golden_dataset)
    print(f"\n🎯 총 {total}개 질문에 대한 Basic RAG 평가를 시작합니다 (top-3 청크)\n")

    chain = rag_prompt | llm

    for idx, item in enumerate(golden_dataset):
        q_id            = item["id"]
        question        = item["question"]
        expected_answer = item["expected_answer"]

        print(f"[{idx+1}/{total}] {q_id} 진행 중...")

        # 1) 관련 청크 상위 3개 검색
        docs = retriever.invoke(question)
        retrieved_years = list(set([d.metadata.get("source_year") for d in docs]))
        retrieved_contexts = [d.page_content for d in docs]
        context = format_docs(docs)

        # 2) RAG 답변 생성
        generated_answer = chain.invoke(
            {"context": context, "question": question}
        ).content.strip()

        results.append({
            "id": q_id,
            "difficulty": item.get("difficulty", "unknown"),
            "source_year": item.get("source_year", ""),
            "retrieved_years": retrieved_years,
            "retrieved_contexts": retrieved_contexts,
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
        })

        print(f"   → 검색 년도: {retrieved_years}")

    # --- 3. 결과 저장 ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 완료! 상세 결과가 '{output_path}'에 저장되었습니다.")
    print(f"   총 {total}개 질문 처리 완료.")


if __name__ == "__main__":
    run_rag_evaluation()
