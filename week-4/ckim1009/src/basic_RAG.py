
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def set_vectorDB(file_path, year, embeddings):
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

    for chunk in chunks:
        chunk.metadata["source_year"] = year
        
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_medical_index")

    print(f"Total Chunks: {len(chunks)}")


    # : 검색 품질 확인 (Top-K=3)
def evaluate_basic_search(dataset, db, k):
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