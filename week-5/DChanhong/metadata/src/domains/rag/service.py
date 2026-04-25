import os
import json
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.core.config import settings, BASE_DIR
from src.utils.pdf_parser import pdf_parser
from src.utils.year_extractor import YearExtractor
from .schemas import QueryRequest, QueryResponse, SourceDocument, FilterInfo


class MetadataRAGService:
    """
    Metadata RAG (옵션 1: Basic + pre-retrieval filter):
      1) 질문에서 년도 추출 (Regex + 상대표현 + cross-year)
      2) ChromaDB similarity_search 호출 시 `filter={"source_year": ...}` 적용
      3) 필터 결과가 너무 적으면 필터 해제 후 재검색 (fallback)
      4) LLM 생성

    BM25·Rerank 없음 (기여도 분리 측정을 위해 pre-filter 만 추가).
    """

    COLLECTION_NAME = "medical_aid_rag_metadata"
    FALLBACK_THRESHOLD_RATIO = 0.5  # 필터 결과가 k 의 절반 이하면 필터 해제

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
        self.year_extractor = YearExtractor(reference_year=settings.REFERENCE_YEAR)

        self.vector_store = None
        self._init_vector_store()

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

    async def run_indexing(self) -> dict:
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
            except Exception as e:
                print(f"[indexing] delete_collection 실패: {e}")
            self._init_vector_store()

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

        # 청크 고유 ID (source + page + start_index)
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "-")
            start = chunk.metadata.get("start_index", 0)
            chunk.metadata["chunk_id"] = f"{src}__p{page}__s{start}"

        self.vector_store.add_documents(chunks)

        return {
            "status": "success",
            "files_processed": pdf_files,
            "total_chunks": len(chunks),
        }

    # ------------------------------------------------------------------
    # 검색 + 생성
    # ------------------------------------------------------------------
    def _build_filter(self, years: List[str]) -> Optional[dict]:
        """Chroma filter 문법으로 변환. 년도 없으면 None."""
        if not years:
            return None
        if len(years) == 1:
            return {"source_year": {"$eq": years[0]}}
        return {"source_year": {"$in": years}}

    async def get_answer(self, request: QueryRequest) -> QueryResponse:
        if not self.vector_store:
            return QueryResponse(
                answer="벡터 스토어가 초기화되지 않았습니다.",
                retrieved_contexts=[],
            )

        # 1) 질문에서 년도 정보 추출
        extracted = self.year_extractor.extract(request.question)
        chroma_filter = self._build_filter(extracted["years"])
        fallback_used = False

        # 2) 필터 적용 검색
        if chroma_filter:
            print(
                f"[metadata] filter={chroma_filter} rationale={extracted['rationale']}"
            )
            docs = self.vector_store.similarity_search(
                request.question, k=request.k, filter=chroma_filter
            )

            # 3) 결과 부족하면 필터 해제 폴백
            min_needed = max(1, int(request.k * self.FALLBACK_THRESHOLD_RATIO))
            if len(docs) < min_needed:
                print(
                    f"[metadata] 필터 결과 부족 ({len(docs)} < {min_needed}) → 필터 해제 재검색"
                )
                docs = self.vector_store.similarity_search(
                    request.question, k=request.k
                )
                fallback_used = True
        else:
            # 년도 신호 없음 → 필터 없이 검색
            print(f"[metadata] 년도 신호 없음 → 전체 검색")
            docs = self.vector_store.similarity_search(
                request.question, k=request.k
            )

        filter_info = FilterInfo(
            applied_years=extracted["years"],
            is_cross_year=extracted["is_cross_year"],
            rationale=extracted["rationale"],
            fallback=fallback_used,
        )
        return await self._generate(request, docs, filter_info)

    async def _generate(
        self,
        request: QueryRequest,
        docs: List[Document],
        filter_info: FilterInfo,
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
            filter_info=filter_info,
        )

    # ------------------------------------------------------------------
    # 평가 (Ragas 입력용 JSONL 생성)
    # ------------------------------------------------------------------
    async def run_evaluation(self) -> dict:
        input_file = (BASE_DIR / "data" / "golden_dataset_v2.jsonl").resolve()
        base_output_dir = (BASE_DIR / "data" / "metadata").resolve()
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
                    "filter_info": resp.filter_info.model_dump() if resp.filter_info else None,
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


rag_service = MetadataRAGService()
