import json
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # 최신 패키지 사용

# 1. 환경 설정
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY가 없습니다.")
    exit()

PDF_PATH = "data/2024 알기 쉬운 의료급여제도.pdf"
GOLDEN_DATA_PATH = "golden_dataset.jsonl"
CHROMA_PERSIST_DIR = "./chroma_db_v2" # 섞이지 않게 새로운 폴더 사용

def run_advanced_rag_pipeline():
    print("🚀 [Step 2 심화] 데이터 파싱 및 청킹 세분화 전략 적용 중...\n")
    
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)

    # ---------------------------------------------------
    # [Step 2] RAG Indexing 파이프라인 (개선버전)
    # ---------------------------------------------------
    # 1. PDFPlumber를 사용하여 표 레이아웃 보존 로딩
    print("1. PDFPlumberLoader로 PDF를 읽어옵니다...")
    loader = PDFPlumberLoader(PDF_PATH)
    docs = loader.load()
    
    # 2. 청킹 세분화 (1000 -> 500으로 축소, 오버랩 100)
    print("2. 문서를 세분화하여 자릅니다 (chunk_size=500)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"✅ [Step 2] 인덱싱 완료: 총 {len(splits)}개의 세분화된 청크 생성!\n")

    # 3. 새로운 Chroma DB 구축
    print("3. 벡터 DB(Chroma) 구축 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print("✅ 벡터 DB 구축 완료\n")

    # ---------------------------------------------------
    # [Step 3] 검색 품질 검증 (재평가)
    # ---------------------------------------------------
    print("--- [Step 3] 개선된 인덱스로 검색 품질(Retrieval Quality) 검증 시작 ---")
    
    golden_data = []
    with open(GOLDEN_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            golden_data.append(json.loads(line.strip()))

    success_count = 0

    for item in golden_data:
        q_id = item['id']
        question = item['question']
        evidence = item['evidence_text']
        difficulty = item['difficulty']

        # Top-3 검색
        search_results = vectorstore.similarity_search(question, k=3)
        
        is_success = False
        retrieved_text_combined = ""
        evidence_compact = evidence.replace(" ", "")
        
        for i, res in enumerate(search_results):
            chunk_content = res.page_content
            retrieved_text_combined += f"\n  [청크 {i+1}] {chunk_content[:150].replace(chr(10), ' ')}..."
            
            # 띄어쓰기 무시하고 일치 여부 확인
            if evidence_compact in chunk_content.replace(" ", ""):
                is_success = True

        status = "✅ 성공" if is_success else "❌ 실패"
        if is_success: success_count += 1
        
        print(f"[{q_id}] 난이도: {difficulty} | 검색 결과: {status}")
        print(f" 🔹 질문: {question}")
        if not is_success:
            print(f" 🔸 기대 근거(Evidence): {evidence}")
            print(f" 🔸 실제 검색된 청크 요약: {retrieved_text_combined}")
        print("-" * 60)

    # 최종 결과 출력
    accuracy = (success_count / len(golden_data)) * 100
    print("========================================")
    print(f"📊 데이터 정제 후 최종 검색 성공률: {success_count}/{len(golden_data)} ({accuracy:.1f}%)")
    print("========================================")

if __name__ == "__main__":
    run_advanced_rag_pipeline()