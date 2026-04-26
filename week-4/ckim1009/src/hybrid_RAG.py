import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from typing import List
from sentence_transformers import CrossEncoder 
from pydantic import PrivateAttr
import os
import json


class CrossEncoderReranker(BaseDocumentCompressor):
    top_n: int = 3
    _model: CrossEncoder = PrivateAttr()

    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", top_n=3):
        super().__init__(top_n=top_n)
        self._model = CrossEncoder(model_name)
        self.top_n = top_n

    def compress_documents(self, documents: List[Document], query: str, callbacks=None):
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in doc_scores[: self.top_n]]
    
# 1. 인덱싱 함수: 메타데이터를 포함하여 청크 생성 및 FAISS 저장
def set_advanced_vectorDB(chunks, embeddings):


    # Vector DB 생성 및 로컬 저장
    # faiss
    # vector_db = FAISS.from_documents(chunks, embeddings)
    # vector_db.save_local("faiss_medical_index")
    
    # chroma
    vector_db = Chroma.from_documents(documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_medical_index")
  

    return vector_db

# 2. Advanced Retriever 설정 함수
def get_advanced_retriever(chunks, vector_db, selected_year):
    """
    selected_year: 특정 년도 필터링이 필요할 경우 (예: "2025")
    """
    # 1) Vector Retriever 설정 (Metadata Filter 포함 가능)
    search_kwargs = {"k": 10}
    if selected_year:
        search_kwargs["filter"] = {"source_year": selected_year}
    
    vector_retriever = vector_db.as_retriever(search_kwargs=search_kwargs)


    # 2) BM25 Retriever 설정 (키워드 검색)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10


    # 3) Hybrid Search (Ensemble)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5] # 벡터와 키워드 비중 조절
    )



    # 4) Re-ranking (Cohere)
    # COHERE_API_KEY가 환경변수에 설정되어 있어야 합니다.
    # compressor = CohereRerank(model="rerank-v3.5", top_n=3)
    
    compressor = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        top_n=3
    )


    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    return advanced_retriever


# 3. 평가 함수: Advanced RAG 성능 측정
def evaluate_advanced_search(chunks, vector_db, golden_dataset):
    success_count = 0
    results_report = []

    for item in golden_dataset:
        query = item["question"]
        evidence = item["evidence_text"]
        source_year = item.get("source_year") 
        
        retriever = get_advanced_retriever(chunks, vector_db, source_year)

        # 실제 검색 수행
        retrieved_docs = retriever.invoke(query)

        # 근거 포함 여부 확인
        is_success = any(evidence in doc.page_content for doc in retrieved_docs)
        


        retrieved_doc = retrieved_docs[0]

        # Source year가 일치하는 것 우선 선택
        for doc in retrieved_docs:
            if  doc.metadata.get("source_year") == source_year:
                retrieved_doc = doc
                break

            
        if is_success:
            success_count += 1
            status = "성공"
        else:
            status = "실패"
        
        result = {
            "id": item["id"],
            "difficulty": item["difficulty"],
            "status": status,
            "source_year": retrieved_doc.metadata.get('source_year'),
            "top_chunk": retrieved_doc.page_content.replace("\n", " ") + "..." if retrieved_docs else "검색 결과 없음"
        }
        print(result['id'],result['source_year'],source_year)
        results_report.append(result)

    accuracy = (success_count / len(golden_dataset)) * 100
    return results_report, accuracy