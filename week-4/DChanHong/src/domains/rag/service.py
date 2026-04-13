import os
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.core.config import settings
from src.utils.pdf_parser import pdf_parser
from .schemas import QueryRequest, QueryResponse, SourceDocument

class RAGService:
    def __init__(self):
        # text-embedding-3-small 모델 사용 (OpenAI)
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0
        )
        self.vector_store = None
        self._init_vector_store()

    def _init_vector_store(self):
        """저장된 벡터 스토어를 로드하거나 초기화합니다."""
        if not os.path.exists(settings.STORAGE_PATH):
            os.makedirs(settings.STORAGE_PATH)
        
        self.vector_store = Chroma(
            persist_directory=settings.STORAGE_PATH,
            embedding_function=self.embeddings,
            collection_name="medical_aid_rag"
        )

    async def run_indexing(self) -> dict:
        """data 폴더의 모든 PDF를 인덱싱합니다."""
        # data 폴더는 DChanHong 바깥에 있는 data 폴더를 가리키도록 설정
        abs_data_path = os.path.abspath(os.path.join(os.getcwd(), settings.DATA_PATH))
        
        if not os.path.exists(abs_data_path):
            return {"error": f"Data path not found at: {abs_data_path}"}

        pdf_files = [f for f in os.listdir(abs_data_path) if f.endswith(".pdf")]
        if not pdf_files:
            return {"error": f"No PDF files found in {abs_data_path}"}
        
        all_documents = []
        for pdf_file in pdf_files:
            file_path = os.path.join(abs_data_path, pdf_file)
            # pdfplumber를 이용한 파싱 (텍스트 + 표 마크다운)
            docs = pdf_parser.parse_pdf(file_path)
            all_documents.extend(docs)

        # 텍스트 분할 (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(all_documents)

        # 벡터 스토어에 추가
        self.vector_store.add_documents(chunks)
        
        return {
            "status": "success",
            "files_processed": pdf_files,
            "total_chunks": len(chunks)
        }

    async def get_answer(self, request: QueryRequest) -> QueryResponse:
        """사용자 질문에 대해 RAG 답변을 생성합니다."""
        if not self.vector_store:
            return QueryResponse(answer="벡터 스토어가 초기화되지 않았습니다. 먼저 인덱싱을 수행해주세요.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        relevant_docs = retriever.invoke(request.question)

        context_parts = []
        source_docs = []
        for doc in relevant_docs:
            source_year = doc.metadata.get("source_year", "unknown")
            source_file = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "-")
            
            context_text = f"[출처: {source_year}년 {source_file}, {page}p]\n{doc.page_content}"
            context_parts.append(context_text)
            
            source_docs.append(SourceDocument(
                content=doc.page_content,
                metadata=doc.metadata
            ))

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 출처 년도가 표시되어 있습니다. 질문이 특정 년도를 묻는 경우 해당 년도의 정보만 사용하세요.
컨텍스트에 없는 내용은 "정보를 찾을 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문: {request.question}

답변:"""

        response = await self.llm.ainvoke(prompt)
        
        return QueryResponse(
            answer=response.content,
            sources=source_docs if request.include_sources else None
        )

rag_service = RAGService()
