#!/usr/bin/env python3
import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ✅ 추가: 허깅페이스 로컬 임베딩
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def build_vector_db(pdf_path: str):
    print(">> PDF 로딩 및 벡터 DB 구축 중...")
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     
        chunk_overlap=50,   
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # ✅ 수정: API 키가 필요 없는 무료 한국어 로컬 오픈소스 모델 사용
    print(">> 로컬 임베딩 모델 다운로드 및 빌드 중 (최초 1회 시간 소요)...")
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# ... (아래 test_golden_dataset 등 나머지 코드는 완전히 동일하게 유지) ...

def test_golden_dataset(vector_db, jsonl_path: str):
    print("\n========== [검색 품질 평가 시작] ==========")
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    success_count = 0
    total_count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            data = json.loads(line)
            question = data['question']
            evidence = data['evidence_text']
            
            print(f"\n[문항 {data['id']}] 난이도: {data['difficulty']}")
            print(f"Q: {question}")
            
            # 검색 수행
            results = retriever.invoke(question)
            
            # 검색된 청크 내용 병합
            retrieved_text = "\n".join([res.page_content for res in results])
            
            # Evidence의 핵심 키워드가 검색 결과에 포함되었는지 대략적인 확인
            # (실제로는 정확한 매칭이 어려우므로, 눈으로 확인하기 위해 출력합니다)
            print(f"-> Expected Evidence: {evidence}")
            print("-> [Top-3 검색 결과 요약]")
            for i, res in enumerate(results):
                # 텍스트가 너무 길면 잘라서 보여줌
                preview = res.page_content.replace('\n', ' ')[:100] + "..."
                print(f"   {i+1}. {preview}")
            print("-" * 50)

if __name__ == "__main__":
    pdf_file_path = "../data/2024 알기 쉬운 의료급여제도.pdf" 
    jsonl_file_path = "golden_dataset.jsonl"
    
    # 1. DB 구축 (한 번만 실행됨)
    db = build_vector_db(pdf_file_path)
    
    # 2. 테스트 수행
    test_golden_dataset(db, jsonl_file_path)