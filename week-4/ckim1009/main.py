import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import pdfplumber
import json
from pathlib import Path
from pydantic import BaseModel, ValidationError, Field
from google import genai


from src.basic_RAG import *
from src.hybrid_RAG import *
from src.llm import *

import warnings
warnings.filterwarnings("ignore")

BASIC_RAG_SAVE_PATH = "output/rag_output/basic_RAG_chunks.json"
ADVANCED_RAG_SAVE_PATH = "output/rag_output/advanced_RAG_chunks_W_meta_filtering.json"

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY 

client = genai.Client(api_key=GEMINI_API_KEY)
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask"
)


def split_documents(file_paths_with_year):
    all_chunks = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    for file_path, year in file_paths_with_year:
        loader = PyPDFLoader(file_path)
        raw_documents = loader.load()
        chunks = text_splitter.split_documents(raw_documents)
        
        # 메타데이터 부여
        for chunk in chunks:
            chunk.metadata["source_year"] = year

        all_chunks.extend(chunks)
    
    print(f"Total Combined Chunks: {len(chunks)}")
    return all_chunks




def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # print(f)
        for line in f:
            try:
                if line.strip():
                    data.append(json.loads(line))
            except:
                print(line)

    return data

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def save_data(file_path, dataset):

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


def eval_basic_rag(doc_2025_path, doc_2026_path):
    # 청크 생성
    chunks = split_documents([[doc_2025_path, 2025], [doc_2026_path,2026]])

    # set_faiss(chunks, embeddings)

    # vector_db = FAISS.load_local("faiss_medical_index", embeddings, allow_dangerous_deserialization=True)

    golden_dataset = load_data('dataset/golden_dataset.jsonl')


    # results_report, hit_rate = evaluate_basic_search(vector_db, golden_dataset, k=3)


    # # 청크 검색 결과 저장
    # save_data(BASIC_RAG_SAVE_PATH, results_report)


    # 청크 검색 결과 로드
    results_report = load_json(BASIC_RAG_SAVE_PATH)

    # 데이터 결합
    for i, result in enumerate(results_report):
        golden_dataset[i]['top_chunk'] = result['top_chunk']

    
    # LLM 질의
    eval_rag_pipeline(client, golden_dataset)

def eval_advanced_rag(doc_2025_path, doc_2026_path):
    # # 청크 생성
    # chunks = split_documents([[doc_2025_path, 2025], [doc_2026_path,2026]])

    # # Chroma 청크 저장
    # # set_advanced_vectorDB(chunks, embeddings)

    
    # # Chroma 로드
    # vector_db = Chroma(persist_directory="chroma_medical_index",
    # embedding_function=embeddings)

    
    # golden dataset 로드
    golden_dataset = load_data('dataset/golden_dataset.jsonl')

    # # vectorDB 청크 검색
    # results_report, accuracy = evaluate_advanced_search(chunks, vector_db, golden_dataset)

    # # 청크 검색 결과 저장
    # save_data(results_report)


    # 청크 검색 결과 로드
    results_report = load_json(ADVANCED_RAG_SAVE_PATH)

    # 데이터 결합
    for i, result in enumerate(results_report):
        golden_dataset[i]['top_chunk'] = result['top_chunk']

    
    # LLM 질의
    eval_rag_pipeline(client, golden_dataset)

def main():
    base_dir = Path(__file__).resolve().parent
    week3_dir = base_dir.parent
    data_dir = week3_dir / 'data'
    doc_2025_path = data_dir / "2025 알기 쉬운 의료급여제도.pdf"
    doc_2026_path = data_dir / "2026 알기 쉬운 의료급여제도.pdf"

    # eval_basic_rag(doc_2025_path, doc_2026_path)
    eval_advanced_rag(doc_2025_path, doc_2026_path)
    

if __name__ == '__main__':
    main()