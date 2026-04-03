import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 환경 설정 및 API 키 로드
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
    exit()

CHROMA_PERSIST_DIR = "./chroma_db"
GOLDEN_DATA_PATH = "golden_dataset.jsonl"

def run_step3_evaluation():
    print("--- [Step 3] 검색 품질(Retrieval Quality) 검증 시작 ---")
    
    # 2. Step 2에서 구축한 기존 벡터 저장소(Chroma) 로드
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    print("✅ 로컬 벡터 저장소(Chroma DB) 로드 완료\n")

    # 3. Golden Dataset 로드
    golden_data = []
    with open(GOLDEN_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            golden_data.append(json.loads(line.strip()))

    success_count = 0
    total_count = len(golden_data)

    # 4. 각 질문별로 Top-K 검색 수행 및 검증
    for item in golden_data:
        q_id = item['id']
        question = item['question']
        evidence = item['evidence_text']
        difficulty = item['difficulty']

        # Top-K (K=3) 검색 수행
        search_results = vectorstore.similarity_search(question, k=3)
        
        # 근거 텍스트 포함 여부 확인 (띄어쓰기로 인한 오류 방지를 위해 공백 제거 후 비교)
        is_success = False
        retrieved_text_combined = ""
        evidence_compact = evidence.replace(" ", "")
        
        for i, res in enumerate(search_results):
            chunk_content = res.page_content
            retrieved_text_combined += f"\n  [청크 {i+1}] {chunk_content[:150].replace(chr(10), ' ')}..." # 150자 요약
            
            if evidence_compact in chunk_content.replace(" ", ""):
                is_success = True

        # 결과 출력
        status = "✅ 성공" if is_success else "❌ 실패"
        if is_success:
            success_count += 1
        
        print(f"[{q_id}] 난이도: {difficulty} | 검색 결과: {status}")
        print(f" 🔹 질문: {question}")
        
        # 실패한 경우에만 어디서 문제가 생겼는지 자세히 출력
        if not is_success:
            print(f" 🔸 기대 근거(Evidence): {evidence}")
            print(f" 🔸 실제 검색된 청크 요약: {retrieved_text_combined}")
        print("-" * 60)

    # 5. 최종 검색 성공률 출력
    accuracy = (success_count / total_count) * 100
    print("========================================")
    print(f"📊 최종 검색 성공률: {success_count}/{total_count} ({accuracy:.1f}%)")
    print("========================================")

if __name__ == "__main__":
    run_step3_evaluation()