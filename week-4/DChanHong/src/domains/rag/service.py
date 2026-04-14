import os
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.core.config import settings
from src.utils.pdf_parser import pdf_parser
from .schemas import QueryRequest, QueryResponse, SourceDocument

# =============================================================================
# [파이썬의 타입 힌트에 대해]
#
# 파이썬은 "동적 타입" 언어입니다.
# -> 변수를 선언할 때 타입을 지정하지 않아도 됩니다.
# -> 같은 변수에 숫자를 넣었다가 문자열을 넣어도 에러가 나지 않습니다.
#
#   예시:
#     x = 10         # x는 정수
#     x = "hello"    # x는 이제 문자열 (에러 없음!)
#
# 반면 Java, TypeScript 같은 "정적 타입" 언어는:
#     int x = 10;
#     x = "hello";   // 컴파일 에러! int에 문자열 넣을 수 없음
#
# 파이썬의 타입 힌트(예: def foo(x: int) -> str:)는:
#   - 실행 시 전혀 강제되지 않습니다 (무시해도 에러 안 남)
#   - "이 함수는 int를 받고 str을 리턴할 예정이에요"라는 문서/안내 역할
#   - IDE(VSCode 등)에서 자동완성, 경고 표시에 활용됨
#   - mypy 같은 별도 도구를 돌려야 타입 검사가 됨
#
# 정리: 파이썬 타입은 "권장사항"이지 "강제사항"이 아닙니다.
# =============================================================================


class RAGService:
    """
    RAG (Retrieval-Augmented Generation) 서비스 클래스

    RAG의 전체 흐름:
    1. [인덱싱] PDF → 텍스트 추출 → 청크로 분할 → 임베딩(벡터화) → 벡터DB 저장
    2. [질의]   사용자 질문 → 임베딩 → 벡터DB에서 유사 문서 검색 → LLM에 전달 → 답변 생성

    이 클래스는 위 두 단계를 각각 run_indexing()과 get_answer() 메서드로 구현합니다.
    """

    def __init__(self):
        """
        RAGService 객체가 생성될 때 자동 실행됩니다.
        여기서 3가지 핵심 도구를 초기화합니다:
          1. embeddings: 텍스트를 벡터(숫자 배열)로 변환하는 도구
          2. llm: 질문에 답변을 생성하는 대규모 언어 모델
          3. vector_store: 벡터를 저장하고 유사도 검색하는 데이터베이스
        """

        # ---------------------------------------------------------------------
        # 1. 임베딩 모델 초기화
        try:
            self.embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,       # "text-embedding-3-small"
                openai_api_key=settings.OPENAI_API_KEY # .env에서 불러온 API 키
            )
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAIEmbeddings: {e}")
            self.embeddings = None

        # 2. LLM 초기화
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=settings.LLM_MODEL,              # "gemini-1.5-flash"
                google_api_key=settings.GEMINI_API_KEY, # .env에서 불러온 API 키
                temperature=0
            )
        except Exception as e:
            print(f"Warning: Failed to initialize ChatGoogleGenerativeAI: {e}")
            self.llm = None

        # ---------------------------------------------------------------------
        # 3. 벡터 스토어 초기화
        # None으로 먼저 선언 후, _init_vector_store()에서 실제 연결
        # None = 아직 값이 없음을 나타내는 파이썬 특수 값
        # ---------------------------------------------------------------------
        self.vector_store = None
        self._init_vector_store()
        # 메서드 이름 앞에 _가 붙으면 "내부용(private)"이라는 관례
        # -> 외부에서 직접 호출하지 말라는 의미 (강제는 아님, 그냥 약속)

    def _init_vector_store(self):
        """
        ChromaDB 벡터 스토어를 로드하거나 새로 초기화합니다.

        ChromaDB란?
        - 벡터(임베딩)를 저장하고, 유사한 벡터를 빠르게 검색하는 데이터베이스
        - persist_directory에 파일로 저장되어 서버를 재시작해도 데이터가 유지됨
        """

        # 저장 경로가 없으면 폴더를 생성
        # os.path.exists(): 해당 경로에 파일/폴더가 있는지 확인 (True/False)
        # os.makedirs(): 폴더를 생성 (중간 경로도 함께 생성)
        if not os.path.exists(settings.STORAGE_PATH):
            os.makedirs(settings.STORAGE_PATH)

        # Chroma 벡터 스토어 생성 또는 기존 것 로드
        self.vector_store = Chroma(
            persist_directory=settings.STORAGE_PATH,  # 데이터 저장 폴더 ("./storage")
            embedding_function=self.embeddings,        # 벡터 변환에 사용할 임베딩 모델
            collection_name="medical_aid_rag"          # 컬렉션 이름 (DB의 테이블 같은 개념)
        )

    async def run_indexing(self) -> dict:
        """
        data 폴더의 모든 PDF를 읽어서 벡터DB에 저장(인덱싱)합니다.
        기존 데이터를 삭제하고 새로 저장합니다.
        """
        # 0단계: 기존 벡터 데이터 삭제 (초기화)
        try:
            if self.vector_store:
                # 기존 컬렉션 삭제
                self.vector_store.delete_collection()
                # 다시 초기화하여 깨끗한 상태로 만듦
                self._init_vector_store()
                print("Existing vector storage cleared.")
        except Exception as e:
            print(f"Warning: Failed to clear existing storage: {e}")

        # ---------------------------------------------------------------------
        # 1단계: data 폴더 경로 확인
        # os.getcwd(): 현재 작업 디렉토리 (Current Working Directory)
        # os.path.join(): 경로를 OS에 맞게 이어붙임 (/ 또는 \)
        # os.path.abspath(): 상대 경로를 절대 경로로 변환
        # 예: os.path.abspath("/a/b/../c") -> "/a/c"
        # ---------------------------------------------------------------------
        abs_data_path = os.path.abspath(os.path.join(os.getcwd(), settings.DATA_PATH))

        if not os.path.exists(abs_data_path):
            return {"error": f"Data path not found at: {abs_data_path}"}

        # ---------------------------------------------------------------------
        # 2단계: PDF 파일 목록 가져오기
        #
        # [리스트 컴프리헨션이란?]
        # 아래 한 줄은 이것과 같습니다:
        #   pdf_files = []
        #   for f in os.listdir(abs_data_path):
        #       if f.endswith(".pdf"):
        #           pdf_files.append(f)
        #
        # -> [결과값 for 변수 in 반복대상 if 조건]
        # -> 조건에 맞는 것만 골라서 리스트를 만드는 축약 문법
        # ---------------------------------------------------------------------
        pdf_files = [f for f in os.listdir(abs_data_path) if f.endswith(".pdf")]
        if not pdf_files:
            return {"error": f"No PDF files found in {abs_data_path}"}

        # ---------------------------------------------------------------------
        # 3단계: 모든 PDF를 파싱하여 Document 리스트로 만듦
        # extend vs append:
        #   append([1,2,3]) -> [[1,2,3]]       (리스트 자체를 하나의 원소로 추가)
        #   extend([1,2,3]) -> [1, 2, 3]        (리스트의 원소들을 각각 추가)
        # ---------------------------------------------------------------------
        all_documents = []
        for pdf_file in pdf_files:
            file_path = os.path.join(abs_data_path, pdf_file)
            # pdf_parser는 pdf_parser.py에서 만든 싱글톤 인스턴스
            docs = pdf_parser.parse_pdf(file_path)
            all_documents.extend(docs)

        # ---------------------------------------------------------------------
        # 4단계: 텍스트 분할 (Chunking)
        #
        # 왜 분할하나?
        # - PDF 한 페이지의 텍스트가 너무 길면 임베딩 품질이 떨어짐
        # - 작은 단위로 나눠야 검색 시 정확한 부분만 찾을 수 있음
        #
        # RecursiveCharacterTextSplitter:
        # - chunk_size=1000: 한 청크의 최대 글자 수
        # - chunk_overlap=200: 앞 청크의 마지막 200자를 다음 청크에도 포함
        #   -> 겹침이 있어야 문장이 잘리는 것을 방지
        #   -> 예: "...문장A 끝부분 | 문장A 끝부분 문장B..."
        # - add_start_index=True: 원본 텍스트에서 몇 번째 글자부터인지 기록
        #
        # "Recursive"인 이유:
        # 단락(\n\n) → 줄바꿈(\n) → 공백( ) → 글자 순서로
        # 최대한 의미 단위를 유지하면서 분할함
        # ---------------------------------------------------------------------
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(all_documents)

        # ---------------------------------------------------------------------
        # 5단계: 벡터 스토어에 저장
        # add_documents()가 내부적으로 하는 일:
        #   1) 각 청크의 텍스트를 임베딩 모델로 벡터로 변환
        #   2) 벡터 + 원본 텍스트 + 메타데이터를 ChromaDB에 저장
        # ---------------------------------------------------------------------
        self.vector_store.add_documents(chunks)

        # 처리 결과를 딕셔너리(dict)로 리턴
        # dict = { "키": 값 } 형태의 데이터 구조 (JavaScript의 Object와 비슷)
        return {
            "status": "success",
            "files_processed": pdf_files,
            "total_chunks": len(chunks)       # len() = 리스트의 길이(개수)
        }

    async def get_answer(self, request: QueryRequest) -> QueryResponse:
        """
        사용자 질문에 대해 RAG 답변을 생성합니다.

        [전체 흐름]
        질문 → 벡터DB에서 유사 문서 검색 → 검색 결과를 프롬프트에 삽입 → LLM이 답변 생성

        [파라미터]
        - request: QueryRequest 객체 (schemas.py에서 정의)
          - request.question: 사용자가 입력한 질문 문자열
          - request.include_sources: 출처 정보를 포함할지 여부 (True/False)
        """

        # 벡터 스토어가 초기화 안 되어있으면 에러 메시지 반환
        if not self.vector_store:
            return QueryResponse(answer="벡터 스토어가 초기화되지 않았습니다. 먼저 인덱싱을 수행해주세요.")

        # ---------------------------------------------------------------------
        # 1단계: 유사 문서 검색 (Retrieval)
        #
        # as_retriever(): 벡터 스토어를 "검색기" 객체로 변환
        # search_kwargs={"k": 10}: 가장 유사한 문서 상위 10개를 가져옴
        #
        # 내부 동작:
        #   1) request.question을 임베딩(벡터)으로 변환
        #   2) 벡터DB에 저장된 모든 벡터와 코사인 유사도 계산
        #   3) 가장 유사한 상위 10개 문서를 리턴
        # ---------------------------------------------------------------------
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        relevant_docs = retriever.invoke(request.question)

        # ---------------------------------------------------------------------
        # 2단계: 검색된 문서들을 프롬프트에 넣을 형태로 가공
        # ---------------------------------------------------------------------
        context_parts = []     # LLM에 전달할 컨텍스트 텍스트 조각들
        source_docs = []       # 응답에 포함할 출처 정보

        for doc in relevant_docs:
            # doc.metadata = {"source": "파일명.pdf", "source_year": "2025", "page": 3}
            # .get("키", 기본값): 해당 키가 있으면 값을 반환, 없으면 기본값 반환
            # -> 딕셔너리["키"]와 비슷하지만, 키가 없어도 에러가 안 남
            source_year = doc.metadata.get("source_year", "unknown")
            source_file = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "-")

            # 검색된 문서 내용에 출처 정보를 붙여서 컨텍스트 구성
            context_text = f"[출처: {source_year}년 {source_file}, {page}p]\n{doc.page_content}"
            context_parts.append(context_text)

            # 응답용 출처 Document 생성
            source_docs.append(SourceDocument(
                content=doc.page_content,
                metadata=doc.metadata
            ))

        # 각 컨텍스트 조각을 "---" 구분선으로 이어붙임
        # "\n\n---\n\n".join(["A", "B", "C"]) -> "A\n\n---\n\nB\n\n---\n\nC"
        context = "\n\n---\n\n".join(context_parts)

        # ---------------------------------------------------------------------
        # 3단계: LLM 프롬프트 구성
        #
        # f-string (f"...{변수}...")을 사용해서
        # 검색된 context와 사용자 question을 프롬프트 템플릿에 삽입
        #
        # 이것이 RAG의 핵심:
        # LLM이 "자기가 아는 것"만으로 답하는 게 아니라,
        # "검색된 문서(context)"를 읽고 그 내용 기반으로 답변함
        # -> 할루시네이션(거짓 답변) 감소
        # -> 최신/도메인 특화 정보에 대한 답변 가능
        # ---------------------------------------------------------------------
        prompt = f"""아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 출처 년도가 표시되어 있습니다. 질문이 특정 년도를 묻는 경우 해당 년도의 정보만 사용하세요.
컨텍스트에 없는 내용은 "정보를 찾을 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문: {request.question}

답변:"""

        # ---------------------------------------------------------------------
        # 4단계: LLM 호출 및 답변 생성
        #
        # await self.llm.ainvoke(prompt)
        # - await: 비동기 함수의 결과를 "기다린다"는 의미
        #   -> LLM API 호출은 네트워크 요청이라 시간이 걸림
        #   -> await으로 기다리는 동안 서버는 다른 요청을 처리할 수 있음
        # - ainvoke: "async invoke"의 약자. invoke의 비동기 버전
        # - response.content: LLM이 생성한 답변 텍스트
        # ---------------------------------------------------------------------
        response = await self.llm.ainvoke(prompt)

        # QueryResponse 객체를 만들어서 리턴
        # response.content가 리스트인 경우를 대비해 문자열로 변환
        answer_text = response.content
        if isinstance(answer_text, list):
            # 리스트 형태인 경우 텍스트 부분만 추출하거나 조인
            answer_text = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in answer_text])

        return QueryResponse(
            answer=str(answer_text),
            sources=source_docs if request.include_sources else None
        )

    async def run_evaluation(self, version: str) -> dict:
        """
        골든 데이터셋을 읽어 전체 질문에 대해 RAG 답변을 생성하고 결과를 파일로 저장합니다.
        
        [저장 경로] data/{version}/{index}/evaluation_results.jsonl
        """
        import json
        from .schemas import EvaluationResult

        input_file = os.path.join("data", "golden_dataset.jsonl")
        base_output_dir = os.path.join("data", version)
        
        # 1. 기본 출력 디렉토리 생성 (예: data/basic)
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)

        # 2. 다음 인덱스 찾기 (0, 1, 2...)
        index = 0
        while os.path.exists(os.path.join(base_output_dir, str(index))):
            index += 1
        
        output_dir = os.path.join(base_output_dir, str(index))
        os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "evaluation_results.jsonl")

        # 3. 골든 데이터셋 확인
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
                
                # RAG 답변 생성 (여기서는 기본 get_answer 호출)
                request = QueryRequest(question=question, include_sources=False)
                response = await self.get_answer(request)
                
                # 결과 딕셔너리 생성 (Pydantic 객체 메서드 충돌 방지 위해 직접 딕셔너리 구성)
                eval_result = {
                    "id": data['id'],
                    "question": question,
                    "expected_answer": data['expected_answer'],
                    "llm_answer": response.answer,
                    "difficulty": data['difficulty'],
                    "source_year": data['source_year']
                }
                
                # 파일에 저장 (json.dumps 사용)
                with open(output_file, 'a', encoding='utf-8') as out_f:
                    out_f.write(json.dumps(eval_result, ensure_ascii=False) + '\n')
                
                processed += 1
                print(f"[{processed}/{total}] Evaluated: {question[:20]}...")
            except Exception as e:
                print(f"Error evaluating question {data.get('id', 'unknown')}: {e}")
                # 에러 발생 시에도 계속 진행하려면 continue, 아니면 에러 발생
                continue

        return {
            "status": "success",
            "endpoint": version,
            "index": index,
            "total": total,
            "processed": processed,
            "output_file": output_file
        }


# =============================================================================
# 싱글톤 인스턴스 생성
# 이 파일이 import될 때 RAGService 객체가 하나 만들어짐
# 다른 파일에서: from .service import rag_service 로 가져다 씀
#
# 주의: 이 시점에 __init__이 실행되므로,
# OpenAI/Gemini API 키가 .env에 있어야 에러가 안 남
# =============================================================================
rag_service = RAGService()
