import os
import json
import cohere
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi

from src.core.config import settings, BASE_DIR
from src.utils.pdf_parser import pdf_parser
from .schemas import QueryRequest, QueryResponse, SourceDocument

# =============================================================================
# BM25용 한국어 토크나이저 (Kiwi 활용)
# =============================================================================
kiwi = Kiwi()

def korean_tokenizer(text: str) -> List[str]:
    """
    한국어 텍스트를 형태소 단위로 토큰화합니다. (BM25 검색 정확도 향상용)
    """
    return [token.form for token in kiwi.tokenize(text)]

class RAGService:
    """
    RAG (Retrieval-Augmented Generation) 서비스 클래스
    
    [Hybrid Search]
    기존 벡터 검색(Dense)과 키워드 검색(Sparse, BM25)을 결합하여 검색 품질을 높입니다.
    
    [Rerank]
    Cohere Rerank를 사용하여 검색된 문서의 순위를 재조정합니다.
    """

    def __init__(self):
        # 1. 임베딩 모델 초기화
        try:
            self.embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY
            )
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAIEmbeddings: {e}")
            self.embeddings = None

        # 2. LLM 초기화
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=settings.LLM_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0
            )
        except Exception as e:
            print(f"Warning: Failed to initialize ChatGoogleGenerativeAI: {e}")
            self.llm = None

        # 3. Cohere Rerank 초기화
        try:
            self.cohere_client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        except Exception as e:
            print(f"Warning: Failed to initialize Cohere Client: {e}")
            self.cohere_client = None

        # 4. 벡터 스토어 및 BM25 초기화
        self.vector_store = None
        self.bm25 = None
        self.bm25_docs = []  # BM25 인덱싱에 사용된 원본 Document 리스트
        
        self._init_vector_store()
        # 벡터 스토어 로드 후 BM25 인덱스도 자동으로 생성 시도
        self._sync_bm25_index()

    def _init_vector_store(self):
        """ChromaDB 벡터 스토어 로드"""
        if not os.path.exists(settings.STORAGE_PATH):
            os.makedirs(settings.STORAGE_PATH)

        self.vector_store = Chroma(
            persist_directory=settings.STORAGE_PATH,
            embedding_function=self.embeddings,
            collection_name="medical_aid_rag"
        )

    def _sync_bm25_index(self):
        """ChromaDB에 저장된 문서를 기반으로 BM25 인덱스를 생성/동기화합니다."""
        try:
            if not self.vector_store:
                return

            # ChromaDB에서 모든 문서 가져오기 (메타데이터 포함)
            all_data = self.vector_store.get()
            if not all_data or not all_data['documents']:
                print("BM25: No documents found in vector store to index.")
                return

            documents = []
            for i in range(len(all_data['documents'])):
                doc = Document(
                    page_content=all_data['documents'][i],
                    metadata=all_data['metadatas'][i] if all_data['metadatas'] else {}
                )
                documents.append(doc)

            self.bm25_docs = documents
            
            # 토큰화 진행 후 BM25 인덱스 생성
            tokenized_corpus = [korean_tokenizer(doc.page_content) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"BM25: Successfully indexed {len(documents)} documents.")
        except Exception as e:
            print(f"Warning: Failed to sync BM25 index: {e}")

    async def run_indexing(self) -> dict:
        """
        data 폴더의 모든 PDF를 읽어서 벡터DB와 BM25 인덱스를 갱신합니다.
        """
        # 0단계: 기존 데이터 삭제
        try:
            if self.vector_store:
                self.vector_store.delete_collection()
                self._init_vector_store()
                self.bm25 = None
                self.bm25_docs = []
                print("Existing storage cleared.")
        except Exception as e:
            print(f"Warning: Failed to clear storage: {e}")

        # 1단계: PDF 목록 및 파싱 (기존 로직 동일)
        abs_data_path = (BASE_DIR / settings.DATA_PATH).resolve()
        if not os.path.exists(abs_data_path):
            return {"error": f"Data path not found at: {abs_data_path}"}

        pdf_files = [f for f in os.listdir(abs_data_path) if f.endswith(".pdf")]
        if not pdf_files:
            return {"error": f"No PDF files found in {abs_data_path}"}

        all_documents = []
        for pdf_file in pdf_files:
            file_path = os.path.join(abs_data_path, pdf_file)
            docs = pdf_parser.parse_pdf(file_path)
            all_documents.extend(docs)

        # 2단계: 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(all_documents)

        # 3단계: 벡터 스토어 저장
        self.vector_store.add_documents(chunks)
        
        # 4단계: BM25 인덱스 즉시 갱신
        self._sync_bm25_index()

        return {
            "status": "success",
            "files_processed": pdf_files,
            "total_chunks": len(chunks),
            "bm25_indexed": len(self.bm25_docs)
        }

    async def get_answer(self, request: QueryRequest) -> QueryResponse:
        """기본 벡터 검색 기반 답변 (기존 기능 유지)"""
        if not self.vector_store:
            return QueryResponse(answer="벡터 스토어가 초기화되지 않았습니다.")

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        relevant_docs = retriever.invoke(request.question)
        
        return await self._generate_answer_from_docs(request, relevant_docs)

    async def get_hybrid_answer(self, request: QueryRequest) -> QueryResponse:
        """벡터 검색 + BM25 검색을 결합한 하이브리드 답변 생성"""
        if not self.vector_store or not self.bm25:
            # BM25가 없으면 기본 검색으로 폴백(Fallback)
            return await self.get_answer(request)

        # 1. 벡터 검색 (Top 20)
        vector_results = self.vector_store.similarity_search(request.question, k=20)
        
        # 2. BM25 검색 (Top 20)
        tokenized_query = korean_tokenizer(request.question)
        bm25_results = self.bm25.get_top_n(tokenized_query, self.bm25_docs, n=20)

        # 3. RRF (Reciprocal Rank Fusion) 결합
        # 점수 = 1 / (rank + k), k=60이 논문 상의 표준값
        k = 60
        rrf_scores = {}

        # 벡터 검색 결과 가중치 부여
        for rank, doc in enumerate(vector_results):
            doc_id = doc.page_content # 내용 자체를 ID로 활용 (정교한 ID가 없을 경우)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + 1 + k)

        # BM25 검색 결과 가중치 부여
        for rank, doc in enumerate(bm25_results):
            doc_id = doc.page_content
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + 1 + k)

        # 4. 점수순 정렬 후 상위 10개 선택
        # 기존 문서 객체를 보존하기 위해 맵 활용
        all_docs_map = {doc.page_content: doc for doc in (vector_results + list(bm25_results))}
        sorted_doc_contents = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        final_docs = [all_docs_map[content] for content in sorted_doc_contents[:10]]

        return await self._generate_answer_from_docs(request, final_docs)

    async def get_reranked_answer(self, request: QueryRequest) -> QueryResponse:
        """하이브리드 검색 후 Cohere Rerank를 적용한 답변 생성"""
        if not self.vector_store or not self.bm25 or not self.cohere_client:
            # 설정이 미비하면 하이브리드 답변으로 폴백
            print("Rerank: Fallback to hybrid search due to missing component.")
            return await self.get_hybrid_answer(request)

        # 1. 하이브리드 후보군 추출 (Best: Dense 20 + Sparse 20)
        vector_results = self.vector_store.similarity_search(request.question, k=20)
        tokenized_query = korean_tokenizer(request.question)
        bm25_results = self.bm25.get_top_n(tokenized_query, self.bm25_docs, n=20)
        
        # 중복 제거 (내용 기준)
        seen_contents = set()
        candidates = []
        for doc in (vector_results + list(bm25_results)):
            if doc.page_content not in seen_contents:
                candidates.append(doc)
                seen_contents.add(doc.page_content)

        # 2. Cohere Rerank 적용 (Best: Top 10)
        try:
            doc_texts = [doc.page_content for doc in candidates]
            response = self.cohere_client.rerank(
                model="rerank-multilingual-v3.0",
                query=request.question,
                documents=doc_texts,
                top_n=10
            )
            
            # 3. 재순위화된 결과 매핑
            final_docs = []
            for result in response.results:
                final_docs.append(candidates[result.index])
            
            print(f"Rerank: Successfully reranked {len(candidates)} docs to top 10.")
        except Exception as e:
            print(f"Warning: Rerank failed: {e}")
            # Rerank 실패 시 하이브리드 RRF 결과 사용
            return await self.get_hybrid_answer(request)

        return await self._generate_answer_from_docs(request, final_docs)

    async def _generate_answer_from_docs(self, request: QueryRequest, docs: List[Document]) -> QueryResponse:
        """검색된 문서들로부터 LLM 답변을 생성하는 공통 로직"""
        context_parts = []
        source_docs = []

        for doc in docs:
            source_year = doc.metadata.get("source_year", "unknown")
            source_file = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "-")

            context_text = f"[출처: {source_year}년 {source_file}, {page}p]\n{doc.page_content}"
            context_parts.append(context_text)
            source_docs.append(SourceDocument(content=doc.page_content, metadata=doc.metadata))

        context = "\n\n---\n\n".join(context_parts)
        prompt = f"""아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 출처 년도가 표시되어 있습니다. 질문이 특정 년도를 묻는 경우 해당 년도의 정보만 사용하세요.
컨텍스트에 없는 내용은 "정보를 찾을 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문: {request.question}

답변:"""

        response = await self.llm.ainvoke(prompt)
        answer_text = response.content
        if isinstance(answer_text, list):
            answer_text = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in answer_text])

        return QueryResponse(
            answer=str(answer_text),
            sources=source_docs if request.include_sources else None
        )

    async def run_evaluation(self, version: str) -> dict:
        """골든 데이터셋 평가 로직 (하이브리드 및 Rerank 대응)"""
        input_file = os.path.join("data", "golden_dataset.jsonl")
        base_output_dir = os.path.join("data", version)
        
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)

        index = 0
        while os.path.exists(os.path.join(base_output_dir, str(index))):
            index += 1
        
        output_dir = os.path.join(base_output_dir, str(index))
        os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "evaluation_results.jsonl")

        if not os.path.exists(input_file):
            return {"error": f"Golden dataset not found at {input_file}"}

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total = len(lines)
        processed = 0

        for line in lines:
            try:
                data = json.loads(line)
                question = data['question']
                
                request = QueryRequest(question=question, include_sources=False)
                
                # version에 따른 검색 방식 선택
                if version == "rerank":
                    response = await self.get_reranked_answer(request)
                elif version == "hybrid":
                    response = await self.get_hybrid_answer(request)
                else:
                    response = await self.get_answer(request)
                
                eval_result = {
                    "id": data['id'],
                    "question": question,
                    "expected_answer": data['expected_answer'],
                    "llm_answer": response.answer,
                    "difficulty": data['difficulty'],
                    "source_year": data['source_year']
                }
                
                with open(output_file, 'a', encoding='utf-8') as out_f:
                    out_f.write(json.dumps(eval_result, ensure_ascii=False) + '\n')
                
                processed += 1
                print(f"[{processed}/{total}] ({version}) Evaluated: {question[:20]}...")
            except Exception as e:
                print(f"Error evaluating question {data.get('id', 'unknown')}: {e}")
                continue

        return {
            "status": "success",
            "endpoint": version,
            "index": index,
            "total": total,
            "processed": processed,
            "output_file": output_file
        }

rag_service = RAGService()
