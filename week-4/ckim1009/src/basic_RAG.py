
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def set_faiss(chunks, embeddings):

        
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_medical_index")

    print(f"Total Chunks: {len(chunks)}")


    # : 검색 품질 확인 (Top-K=3)
def evaluate_basic_search(vector_db, golden_dataset, k):
    success_count = 0
    results_report = []

    for item in golden_dataset:
        query = item["question"]
        evidence = item["evidence_text"]
        
        # 검색 수행
        retrieved_docs = vector_db.similarity_search(query, k=k)
        
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
            "source_year": retrieved_docs[0].metadata.get('source_year'),
            "top_chunk": retrieved_docs[0].page_content.replace("\n", " ") + "..."
        }

        results_report.append(result)

    return results_report, (success_count / len(golden_dataset)) * 100