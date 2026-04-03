import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 환경 설정
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def run_final_indexing(upstage_json_path):
    # --- Step 1: Upstage 결과 로딩 및 기본 청킹 ---
    with open(upstage_json_path, "r", encoding="utf-8") as f:
        upstage_data = json.load(f)

    raw_documents = []
    
    # Upstage의 elements를 순회하며 Document 객체 생성
    # 이미 Upstage가 표(table), 문단(paragraph)을 나눠놨으므로 이게 1차 청킹입니다.
    for element in upstage_data.get("elements", []):
        category = element.get("category")
        content_html = element.get("content", {}).get("html", "")
        page_num = element.get("page")
        
        if content_html:
            # 표 데이터는 마크다운에서 검색이 잘 되도록 간단한 태그 보정을 할 수 있습니다.
            # 여기서는 원본 HTML 구조를 유지하되 메타데이터를 강화합니다.
            doc = Document(
                page_content=content_html,
                metadata={
                    "page": page_num,
                    "category": category,
                    "source": "upstage_parsed"
                }
            )
            raw_documents.append(doc)

    # --- Step 2: 세부 청킹 (Text Splitting) ---
    # 표는 쪼개지 않고, 너무 긴 일반 텍스트 문단만 적절히 자릅니다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    final_docs = text_splitter.split_documents(raw_documents)
    print(f"📦 총 {len(final_docs)}개의 의미 있는 청크가 생성되었습니다.")

    # --- Step 3 & 4: 임베딩 및 벡터 저장소 저장 ---
    print("🧠 임베딩 모델 로드 중 (gemini-embedding-001)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    print("💾 FAISS 벡터 저장소 생성 및 로컬 저장 중...")
    vectorstore = FAISS.from_documents(final_docs, embeddings)
    
    # 나중에 evaluate.py에서 쓸 이름과 동일하게 저장합니다.
    index_name = "medical_upstage_index"
    vectorstore.save_local(index_name)
    
    print(f"✅ 저장 완료! 인덱스 폴더명: {index_name}")

if __name__ == "__main__":
    # 아까 저장했던 JSON 파일 경로를 넣으세요.
    run_final_indexing("page4_result.json")