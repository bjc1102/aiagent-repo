import os
import json
from typing import List

import cohere
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
# BM25 용 한국어 형태소 토크나이저
# =============================================================================
kiwi = Kiwi()


def korean_tokenizer(text: str) -> List[str]:
    return [token.form for token in kiwi.tokenize(text)]


class RerankRAGService:
    """
    Rerank RAG:
      1) Dense retrieval  (ChromaDB) Top 20
      2) Sparse retrieval (BM25)     Top 20
      3) 두 결과 union → 중복 제거 (후보군 40 이하)
      4) Cohere Rerank → Top k
      5) LLM 생성

    RRF 대신 Rerank 를 쓰는 이유:
    - RRF 는 "등장 순위"만 봄 (상대적)
    - Rerank 는 cross-encoder 가 질문·문서를 함께 읽어 관련도 점수를 냄 (절대적)
      → 일반적으로 품질 ↑, 비용 ↑ (Cohere API 호출 필요)
    """

    COLLECTION_NAME = "medical_aid_rag_rerank"

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0,
        )

        # Cohere rerank client
        try:
            self.cohere_client = (
                cohere.ClientV2(api_key=settings.COHERE_API_KEY)
                if settings.COHERE_API_KEY
                else None
            )
            if self.cohere_client is None:
                print("[rerank] COHERE_API_KEY 없음 → Hybrid(RRF) 폴백 모드")
        except Exception as e:
            print(f"[rerank] Cohere client 초기화 실패: {e}")
            self.cohere_client = None

        self.vector_store = None
        self.bm25: BM25Okapi | None = None
        self.bm25_docs: List[Document] = []

        self._init_vector_store()
        self._sync_bm25_index()

    # ------------------------------------------------------------------
    # 초기화 / 인덱싱
    # ------------------------------------------------------------------
    def _init_vector_store(self):
        storage_path = str((BASE_DIR / settings.STORAGE_PATH).resolve())
        os.makedirs(storage_path, exist_ok=True)
        self.vector_store = Chroma(
            persist_directory=storage_path,
            embedding_function=self.embeddings,
            collection_name=self.COLLECTION_NAME,
        )

    def _sync_bm25_index(self):
        try:
            if not self.vector_store:
                return
            all_data = self.vector_store.get()
            if not all_data or not all_data.get("documents"):
                print("[bm25] 벡터 스토어에 인덱싱된 문서 없음")
                return

            documents = []
            for i in range(len(all_data["documents"])):
                documents.append(
                    Document(
                        page_content=all_data["documents"][i],
                        metadata=all_data["metadatas"][i]
                        if all_data["metadatas"]
                        else {},
                    )
                )
            self.bm25_docs = documents
            tokenized_corpus = [korean_tokenizer(d.page_content) for d in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"[bm25] {len(documents)} 청크 인덱싱 완료")
        except Exception as e:
            print(f"[bm25] 인덱스 동기화 실패: {e}")

    async def run_indexing(self) -> dict:
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
            except Exception as e:
                print(f"[indexing] delete_collection 실패: {e}")
            self._init_vector_store()
            self.bm25 = None
            self.bm25_docs = []

        abs_data_path = (BASE_DIR / settings.DATA_PATH).resolve()
        if not os.path.exists(abs_data_path):
            return {"error": f"data 경로 없음: {abs_data_path}"}

        pdf_files = [f for f in os.listdir(abs_data_path) if f.endswith(".pdf")]
        if not pdf_files:
            return {"error": f"PDF 없음: {abs_data_path}"}

        all_documents: List[Document] = []
        for pdf_file in pdf_files:
            docs = pdf_parser.parse_pdf(os.path.join(abs_data_path, pdf_file))
            all_documents.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        chunks = splitter.split_documents(all_documents)

        # 청크 고유 ID 부여 (source + page + start_index)
        # 다년도 PDF 에서 같은 문장이 있어도 chunk_id 로 개별 식별되도록 함.
        # Rerank 단계의 중복 제거가 page_content 기반이면 2025/2026 청크가
        # 한 항목으로 합쳐져 한쪽이 사라짐 → 년도 혼동 분석 오염.
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "-")
            start = chunk.metadata.get("start_index", 0)
            chunk.metadata["chunk_id"] = f"{src}__p{page}__s{start}"

        self.vector_store.add_documents(chunks)
        self._sync_bm25_index()

        return {
            "status": "success",
            "files_processed": pdf_files,
            "total_chunks": len(chunks),
            "bm25_indexed": len(self.bm25_docs),
        }

    # ------------------------------------------------------------------
    # 검색 + 생성
    # ------------------------------------------------------------------
    async def get_answer(self, request: QueryRequest) -> QueryResponse:
        if not self.vector_store:
            return QueryResponse(
                answer="벡터 스토어가 초기화되지 않았습니다.",
                retrieved_contexts=[],
            )

        # Cohere 없거나 BM25 없으면 Hybrid(RRF) 로 폴백
        if not self.bm25 or not self.cohere_client:
            print("[rerank] 필수 컴포넌트 부족 → Hybrid RRF 폴백")
            return await self._answer_with_rrf(request)

        # 1) Dense + 2) Sparse 후보 수집
        vector_results = self.vector_store.similarity_search(
            request.question, k=request.candidates
        )
        tokenized_query = korean_tokenizer(request.question)
        bm25_results = self.bm25.get_top_n(
            tokenized_query, self.bm25_docs, n=request.candidates
        )

        # 3) 중복 제거 (chunk_id 기준; 없으면 page_content 폴백)
        seen = set()
        candidates: List[Document] = []
        for doc in list(vector_results) + list(bm25_results):
            cid = doc.metadata.get("chunk_id") or doc.page_content
            if cid not in seen:
                candidates.append(doc)
                seen.add(cid)

        # 4) Cohere Rerank
        try:
            response = self.cohere_client.rerank(
                model=settings.RERANK_MODEL,
                query=request.question,
                documents=[d.page_content for d in candidates],
                top_n=request.k,
            )
            final_docs = [candidates[r.index] for r in response.results]
            print(
                f"[rerank] {len(candidates)} 후보 → 상위 {len(final_docs)} 선택"
            )
        except Exception as e:
            print(f"[rerank] Cohere 호출 실패: {e} → Hybrid RRF 폴백")
            return await self._answer_with_rrf(request)

        return await self._generate(request, final_docs)

    async def _answer_with_rrf(self, request: QueryRequest) -> QueryResponse:
        """Cohere 불가 시 폴백: Dense + BM25 + RRF"""
        if not self.bm25:
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": request.k}
            )
            docs = retriever.invoke(request.question)
            return await self._generate(request, docs)

        vector_results = self.vector_store.similarity_search(
            request.question, k=request.candidates
        )
        tokenized_query = korean_tokenizer(request.question)
        bm25_results = self.bm25.get_top_n(
            tokenized_query, self.bm25_docs, n=request.candidates
        )

        # chunk_id 기준 (폴백: page_content) — 다년도 청크 병합 방지
        def _key(d: Document) -> str:
            return d.metadata.get("chunk_id") or d.page_content

        k_const = 60
        scores: dict[str, float] = {}
        for rank, doc in enumerate(vector_results):
            k = _key(doc)
            scores[k] = scores.get(k, 0) + 1 / (rank + 1 + k_const)
        for rank, doc in enumerate(bm25_results):
            k = _key(doc)
            scores[k] = scores.get(k, 0) + 1 / (rank + 1 + k_const)

        docs_map = {
            _key(d): d for d in list(vector_results) + list(bm25_results)
        }
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        final_docs = [docs_map[k] for k in sorted_keys[: request.k]]
        return await self._generate(request, final_docs)

    async def _generate(
        self, request: QueryRequest, docs: List[Document]
    ) -> QueryResponse:
        context_parts = []
        sources = []
        retrieved_contexts = []

        for doc in docs:
            year = doc.metadata.get("source_year", "unknown")
            file = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "-")

            context_parts.append(
                f"[출처: {year}년 {file}, {page}p]\n{doc.page_content}"
            )
            retrieved_contexts.append(doc.page_content)
            sources.append(
                SourceDocument(content=doc.page_content, metadata=doc.metadata)
            )

        context = "\n\n---\n\n".join(context_parts)
        prompt = (
            "아래 컨텍스트를 바탕으로 질문에 답하세요.\n"
            "각 컨텍스트에는 출처 년도가 표시되어 있습니다. "
            "질문이 특정 년도를 묻는 경우 해당 년도의 정보만 사용하세요.\n"
            "컨텍스트에 없는 내용은 \"정보를 찾을 수 없습니다\"라고 답하세요.\n\n"
            f"컨텍스트:\n{context}\n\n"
            f"질문: {request.question}\n\n답변:"
        )

        response = await self.llm.ainvoke(prompt)
        answer_text = response.content
        if isinstance(answer_text, list):
            answer_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in answer_text
            )

        return QueryResponse(
            answer=str(answer_text),
            retrieved_contexts=retrieved_contexts,
            sources=sources if request.include_sources else None,
        )

    # ------------------------------------------------------------------
    # 평가 (Ragas 입력용 JSONL 생성)
    # ------------------------------------------------------------------
    async def run_evaluation(self) -> dict:
        input_file = (BASE_DIR / "data" / "golden_dataset_v2.jsonl").resolve()
        base_output_dir = (BASE_DIR / "data" / "rerank").resolve()
        os.makedirs(base_output_dir, exist_ok=True)

        index = 0
        while (base_output_dir / str(index)).exists():
            index += 1
        output_dir = base_output_dir / str(index)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "evaluation_results.jsonl"

        if not input_file.exists():
            return {"error": f"golden dataset 없음: {input_file}"}

        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        processed = 0
        for line in lines:
            try:
                data = json.loads(line)
                req = QueryRequest(
                    question=data["question"], include_sources=False
                )
                resp = await self.get_answer(req)

                out = {
                    "question": data["question"],
                    "ground_truth": data.get("ground_truth", ""),
                    "ground_truth_contexts": data.get("ground_truth_contexts", []),
                    "response": resp.answer,
                    "retrieved_contexts": resp.retrieved_contexts,
                    "difficulty": data.get("difficulty"),
                    "source_year": data.get("source_year"),
                }
                with open(output_file, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                processed += 1
                print(f"[{processed}/{len(lines)}] {data['question'][:30]}...")
            except Exception as e:
                print(f"평가 실패: {e}")
                continue

        return {
            "status": "success",
            "index": index,
            "total": len(lines),
            "processed": processed,
            "output_file": str(output_file),
        }


rag_service = RerankRAGService()
