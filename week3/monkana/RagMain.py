import json
import re
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils import CHROMA_DIR, GOLDEN_PATH, load_and_split_pdf

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_langchain_db"

model = ChatOpenAI(model="gpt-5.2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 기존 벡터 DB 삭제
if CHROMA_DIR.exists():
    shutil.rmtree(CHROMA_DIR)

# 문서 분할
all_splits = load_and_split_pdf(
    max_characters=1000,
    combine_text_under_n_chars=300,
    new_after_n_chars=800,
)

# 벡터 스토어 생성
vector_store = Chroma(
    collection_name="medical_pdf",
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
)

# 문서 적재
vector_store.add_documents(all_splits)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text


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

    if hasattr(response, "content"):
        return str(response.content).strip()
    return str(response).strip()


def check_retrieval_success(retrieved_docs, evidence_text: str):
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


def print_eval_table(rows, total_success, total_count):
    headers = ["질문 ID", "난이도", "검색 결과", "검색된 청크 요약"]

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


def load_golden_dataset():
    golden_rows = []
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                golden_rows.append(json.loads(line))
    return golden_rows


if __name__ == "__main__":
    golden_rows = load_golden_dataset()

    eval_rows = []
    retrieval_success_count = 0

    for item in golden_rows:
        qid = item["id"]
        question = item["question"]
        expected_answer = item["expected_answer"]
        evidence_text = item["evidence_text"]
        difficulty = item.get("difficulty", "")

        retrieved_docs = vector_store.similarity_search(question, k=2)

        retrieval_success, matched_chunks, evidence_tokens, found_tokens = check_retrieval_success(
            retrieved_docs, evidence_text
        )

        if retrieval_success:
            retrieval_success_count += 1

        predicted_answer = generate_answer_from_context(question, retrieved_docs)
        answer_match = normalize_text(predicted_answer) == normalize_text(expected_answer)

        search_result = "성공" if retrieval_success else "실패"

        chunk_summary = make_chunk_summary(
            retrieved_docs,
            matched_chunks,
            evidence_tokens,
            found_tokens,
        )
        chunk_summary += (
            f" || 예상답: {expected_answer} / 모델답: {predicted_answer} / 답변일치: "
            f"{'O' if answer_match else 'X'}"
        )

        eval_rows.append([qid, difficulty, search_result, chunk_summary])

    print_eval_table(eval_rows, retrieval_success_count, len(golden_rows))