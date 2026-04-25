import os
import json
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.core.config import settings, BASE_DIR
from src.utils.pdf_parser import pdf_parser
from .schemas import QueryRequest, QueryResponse, SourceDocument


class BasicRAGService:
    """
    Basic RAG: Dense Retrieval (ChromaDB) + LLM 생성 만 수행.
    Hybrid / Rerank / Metadata 버전은 별도 폴더에서 구현.
    """

    COLLECTION_NAME = "medical_aid_rag_basic"

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

        self.vector_store = None
        self._init_vector_store()

    def _init_vector_store(self):
        storage_path = str((BASE_DIR / settings.STORAGE_PATH).resolve())
        os.makedirs(storage_path, exist_ok=True)

        self.vector_store = Chroma(
            persist_directory=storage_path,
            embedding_function=self.embeddings,
            collection_name=self.COLLECTION_NAME,
        )

    async def run_indexing(self) -> dict:
        """data/ 폴더의 모든 PDF 를 읽어서 벡터 DB 를 새로 만듭니다."""
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
            except Exception as e:
                print(f"[indexing] delete_collection 실패: {e}")
            self._init_vector_store()

        abs_data_path = (BASE_DIR / settings.DATA_PATH).resolve()
        if not os.path.exists(abs_data_path):
            return {"error": f"data 경로를 찾을 수 없습니다: {abs_data_path}"}

        pdf_files = [f for f in os.listdir(abs_data_path) if f.endswith(".pdf")]
        if not pdf_files:
            return {"error": f"PDF 가 없습니다: {abs_data_path}"}

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
        # 다년도 PDF 에서 내용이 동일해도 년도별로 구분되도록 하기 위함
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

    async def get_answer(self, request: QueryRequest) -> QueryResponse:
        if not self.vector_store:
            return QueryResponse(
                answer="벡터 스토어가 초기화되지 않았습니다.",
                retrieved_contexts=[],
            )

        retriever = self.vector_store.as_retriever(search_kwargs={"k": request.k})
        relevant_docs = retriever.invoke(request.question)
        return await self._generate(request, relevant_docs)

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

    async def run_evaluation(self) -> dict:
        """golden_dataset_v2.jsonl 을 읽어 Basic RAG 로 전체 답변 생성 → Ragas 입력용 파일 저장"""
        input_file = (BASE_DIR / "data" / "golden_dataset_v2.jsonl").resolve()
        base_output_dir = (BASE_DIR / "data" / "basic").resolve()
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


rag_service = BasicRAGService()
