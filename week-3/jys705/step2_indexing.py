import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 환경 변수 로드
load_dotenv()

# 1. PDF 로딩 (Step 2-1)
# 로더 선택 및 문서 로딩
loader = PyPDFLoader("data/2024 알기 쉬운 의료급여제도.pdf")
documents = loader.load()

# 2. 청킹 (Step 2-1)
# Text Splitter 선택 및 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True # 추적을 위해 인덱스 추가
)
splits = text_splitter.split_documents(documents)

# 3. 임베딩 및 벡터 저장소 저장 (Step 2-2)
# 임베딩 모델 및 벡터 저장소 선택
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)

print(f"--- 인덱싱 결과 기록 ---")
print(f"총 생성 청크 수: {len(splits)}")