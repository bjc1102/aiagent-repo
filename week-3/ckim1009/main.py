import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import pdfplumber
import json
from pathlib import Path

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# 환경 변수 설정 (Google API Key 필요)
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY 

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def save_vectordb(chunks): 
    # FAISS 벡터 저장소 생성 및 로컬 저장
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_medical_index")

    print(f"Total Chunks: {len(chunks)}")


def parse_pdf(file_path):
    all_docs = []
    
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            
            # 표 추출 설정 (병합된 셀 값을 모든 칸에 채우도록 설정)
            table_settings = {
                "vertical_strategy": "lines",   # 선(Line)을 기준으로 열을 나눔
                "horizontal_strategy": "lines", # 선을 기준으로 행을 나눔
                "snap_tolerance": 3,            # 3픽셀 이내의 떨어진 선은 붙여서 인식
                "join_tolerance": 3,            # 선들이 교차하지 않아도 가깝다면 교차로 간주
            }
            # 1. 해당 페이지의 표(Table) 추출
            tables = page.extract_tables(table_settings=table_settings)

            for table in tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                # 표를 LLM이 이해하기 쉬운 Markdown 형식으로 변환
                md_table = df.to_markdown(index=False)
                all_docs.append(md_table)
            
            # 2. 텍스트 추출 (표 제외 영역만 뽑거나 전체 정제)
            text = page.extract_text()
            if text:
                # Q&A 패턴 등을 로직으로 분리 (예: Q로 시작하는 줄부터 다음 Q 전까지)
                all_docs.append(text)
    
    save_vectordb(all_docs)

def set_vectorDB(file_path):
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()

    # Text Splitter 설정
    # 본인부담률 표는 내용이 조밀하므로 chunk_size를 너무 작지 않게 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_documents)





# Step 3: 검색 품질 확인 (Top-K=3)
def evaluate_retrieval(db, dataset, k):
    success_count = 0
    results_report = []

    for item in dataset:
        query = item["question"]
        evidence = item["evidence_text"]
        
        # 검색 수행
        retrieved_docs = db.similarity_search(query, k=k)
        
        # print(retrieved_docs)

        # 근거 포함 여부 확인
        is_success = any(evidence in doc.page_content for doc in retrieved_docs)
        if is_success:
            success_count += 1
            status = "성공"
        else:
            status = "실패"
        
        result = {
            "id": item["id"],
            "difficulty": item["difficulty"],
            "status": status,
            "top_chunk": retrieved_docs[0].page_content[:30].replace("\n", " ") + "..."
        }
        print(result)
        results_report.append(result)

    return results_report, (success_count / len(dataset)) * 100

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    base_dir = Path(__file__).resolve().parent
    week3_dir = base_dir.parent
    data_dir = week3_dir / 'data'
    file_path = data_dir / "2024 알기 쉬운 의료급여제도.pdf"

    set_vectorDB(str(file_path))

    k=3
    
    # # 실행
    vector_db = FAISS.load_local("faiss_medical_index", embeddings, allow_dangerous_deserialization=True)

    docstore = vector_db.docstore
    all_docs = list(docstore._dict.values())
    # print(f"전체 청크 개수: {len(all_docs)}")


    # for i, doc in enumerate(all_docs):
    #     print(f'✅{i}\n')
    #     print(f"샘플 청크 내용: {doc.page_content[:100]}...")
    #     print(f"메타데이터: {all_docs[0].metadata}")

    golden_dataset = load_data('dataset/golden_dataset.jsonl')

    # 평가 실행
    eval_report, hit_rate = evaluate_retrieval(vector_db, golden_dataset, k)

    # print(eval_report)
    print(hit_rate)

if __name__ == '__main__':
    main()