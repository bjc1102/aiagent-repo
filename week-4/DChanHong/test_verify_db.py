import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from src.core.config import settings

def verify():
    print("### Vector DB Verification Start ###")
    
    # 1. 임베딩 및 DB 로드
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY
    )
    
    vector_store = Chroma(
        persist_directory=settings.STORAGE_PATH,
        embedding_function=embeddings,
        collection_name="medical_aid_rag"
    )
    
    # 2. 전체 데이터 개수 확인
    collection = vector_store._collection
    count = collection.count()
    print(f"Total Chunks in DB: {count}")
    
    # 3. 데이터 샘플 조회 (최초 5개)
    results = collection.get(limit=5)
    print("\n--- Sample Chunks ---")
    for i in range(len(results['documents'])):
        doc = results['documents'][i]
        meta = results['metadatas'][i]
        print(f"[{i+1}] Source: {meta.get('source')} | Year: {meta.get('source_year')}")
        print(f"Content Preview: {doc[:100]}...")
        print("-" * 30)
    
    # 4. 년도별 통계 확인
    all_metas = collection.get(include=['metadatas'])['metadatas']
    years = [m.get('source_year') for m in all_metas]
    from collections import Counter
    stats = Counter(years)
    print("\n--- Year Distribution ---")
    for year, c in stats.items():
        print(f"Year {year}: {c} chunks")

if __name__ == "__main__":
    verify()
