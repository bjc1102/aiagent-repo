from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
import json
import re
from pathlib import Path

#채팅 모델
model = ChatOpenAI(model="gpt-5.2")

#임베딩 모델
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

#백터 스토어 - 크로마
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
#문서 로드
pdf_filepath = "./2024 알기 쉬운 의료급여제도.pdf"

loader = PyPDFLoader(pdf_filepath, mode="page")
page_docs = loader.load()

#문서 분할(청킹)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=300,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(page_docs)

#문서 분할(저장) - Chroma vector db 에 청크 원문과 청크 백터 그리고 메타데이터 자장
document_ids = vector_store.add_documents(documents=all_splits)

#RAG 에이전트 - 백터 스토어 검색 함수를 LangChain 에이전트가 호출할 수 있도록 도구로서 등록
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):   #사용자 질의 받으면, 
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2) #query를 백터 스토어와 연결된 임베딩 모델로 백터화 후 저장된 문서 백터들과 유사도 계산해서 상위 2개 반환
    serialized = "\n\n".join(   #찾은 쿼리 값들을 LLM이 읽기 쉽게 문자열 형태로 변환
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs #LLM에게 바로 넣기 좋은 문자열 형태의 검색 결과, 원본 Document 객체

import json
import re
from pathlib import Path


# -----------------------------
# 1) Golden Dataset 로드
# -----------------------------
golden_path = "./goldenDataset.jsonl"

golden_rows = []
with open(golden_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            golden_rows.append(json.loads(line))


# -----------------------------
# 2) 문자열 정규화 함수
#    - 공백 제거
#    - 소문자화
#    - 필요하면 쉼표 제거
# -----------------------------
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text


# -----------------------------
# 3) LLM 답변 생성
#    - 검색된 context만 근거로 짧게 답하게 함
# -----------------------------
def generate_answer_from_context(question: str, retrieved_docs):
    context_text = "\n\n".join(
        [
            f"[문서 {i+1}]"
            f"\nsource: {doc.metadata}"
            f"\ncontent: {doc.page_content}"
            for i, doc in enumerate(retrieved_docs)
        ]
    )

    prompt = f"""
너는 검색된 문맥만 근거로 답하는 QA 평가기다.

규칙:
- 반드시 아래 [검색 문맥] 안에서만 답해라.
- 질문에 대한 최종 답만 아주 짧게 출력해라.
- 설명, 근거, 부가 문장 없이 답만 출력해라.
- 문맥에 없으면 "모름"이라고 출력해라.

[질문]
{question}

[검색 문맥]
{context_text}
""".strip()

    response = model.invoke(prompt)

    # ChatOpenAI 응답에서 텍스트 추출
    if hasattr(response, "content"):
        return str(response.content).strip()
    return str(response).strip()


# -----------------------------
# 4) evidence_text를 ',' 기준으로 나눈 뒤
#    각 토큰이 Top-K 청크에 포함되는지 검사
#    - 전부 포함되면 성공
#    - 하나라도 없으면 실패
# -----------------------------
def check_retrieval_success(retrieved_docs, evidence_text: str):
    # evidence_text를 ',' 기준으로 분리
    evidence_tokens = [
        normalize_text(token)
        for token in evidence_text.split(",")
        if normalize_text(token)
    ]

    matched_chunks = []
    found_tokens = set()

    for idx, doc in enumerate(retrieved_docs, start=1):
        chunk_text = doc.page_content or ""
        chunk_norm = normalize_text(chunk_text)

        chunk_matched_tokens = []

        for token in evidence_tokens:
            if token in chunk_norm:
                found_tokens.add(token)
                chunk_matched_tokens.append(token)

        if chunk_matched_tokens:
            matched_chunks.append(
                {
                    "chunk_index": idx,
                    "doc": doc,
                    "matched_tokens": chunk_matched_tokens,
                }
            )

    success = len(evidence_tokens) > 0 and all(token in found_tokens for token in evidence_tokens)

    return success, matched_chunks, evidence_tokens, sorted(found_tokens)


# -----------------------------
# 5) 청크 요약 문자열
# -----------------------------
def make_chunk_summary(retrieved_docs, matched_chunks, evidence_tokens, found_tokens, max_len=90):
    summaries = []

    matched_map = {
        item["chunk_index"]: item["matched_tokens"]
        for item in matched_chunks
    }

    for idx, doc in enumerate(retrieved_docs, start=1):
        raw = (doc.page_content or "").replace("\n", " ").strip()
        short = raw[:max_len] + ("..." if len(raw) > max_len else "")

        if idx in matched_map:
            token_str = ", ".join(matched_map[idx])
            prefix = f"[청크{idx}](매칭:{token_str})"
        else:
            prefix = f"[청크{idx}]"

        summaries.append(f"{prefix} {short}")

    missing_tokens = [token for token in evidence_tokens if token not in found_tokens]

    summaries.append(f"|| evidence 토큰: {evidence_tokens}")
    summaries.append(f"|| 찾은 토큰: {found_tokens}")
    summaries.append(f"|| 누락 토큰: {missing_tokens}")

    return " | ".join(summaries)


# -----------------------------
# 6) 표 출력용 함수
# -----------------------------
def print_eval_table(rows, total_success, total_count):
    headers = ["질문 ID", "난이도", "검색 결과", "검색된 청크 요약"]

    # 각 컬럼 폭 계산
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    def hline():
        print("+" + "+".join("-" * (w + 2) for w in col_widths) + "+")

    def print_row(values):
        print(
            "| "
            + " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(values))
            + " |"
        )

    hline()
    print_row(headers)
    hline()

    for row in rows:
        print_row(row)

    hline()
    print_row(["검색 성공률", "", f"{total_success}/{total_count}", ""])
    hline()


# -----------------------------
# 7) 평가 실행
#    - 검색 성공/실패
#    - expected_answer 일치/불일치
#    - 표 형태 출력
# -----------------------------
eval_rows = []
retrieval_success_count = 0

for item in golden_rows:
    qid = item["id"]
    question = item["question"]
    expected_answer = item["expected_answer"]
    evidence_text = item["evidence_text"]
    difficulty = item.get("difficulty", "")

    # Top-K 검색 (현재 코드와 동일하게 k=2)
    retrieved_docs = vector_store.similarity_search(question, k=2)

    # 검색 성공 판정
    retrieval_success, matched_chunks, evidence_tokens, found_tokens = check_retrieval_success(
    retrieved_docs, evidence_text
)

    if retrieval_success:
        retrieval_success_count += 1

    # 답변 생성 및 정답 비교
    predicted_answer = generate_answer_from_context(question, retrieved_docs)

    answer_match = normalize_text(predicted_answer) == normalize_text(expected_answer)

    # 검색 결과 컬럼
    # 표는 이미지처럼 단순하게 두되, 답변 일치 여부도 같이 표시
    if retrieval_success:
        search_result = "성공"
    else:
        search_result = "실패"

    # 청크 요약 + 답변 비교 결과 같이 표시
    chunk_summary = make_chunk_summary(
    retrieved_docs,
    matched_chunks,
    evidence_tokens,
    found_tokens
    )
    chunk_summary += f" || 예상답: {expected_answer} / 모델답: {predicted_answer} / 답변일치: {'O' if answer_match else 'X'}"

    eval_rows.append([qid, difficulty, search_result, chunk_summary])

# 최종 출력
print_eval_table(eval_rows, retrieval_success_count, len(golden_rows))

