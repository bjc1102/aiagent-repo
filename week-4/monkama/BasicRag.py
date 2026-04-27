import json
import os
import re
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"


def reset_chroma_dir():
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

REBUILD_DB = True


def chunk_subject(chunk: dict) -> str:
    return chunk.get("subject", chunk.get("content", ""))


def chunk_parent_subject(chunk: dict) -> str:
    return chunk.get("parent_subject", chunk.get("parent_section", ""))


def table_cell(text: str, limit: int = 80) -> str:
    text = str(text).replace("\n", " ").replace("|", "\\|").strip()
    return text[:limit]


def get_embeddings():
    return OpenAIEmbeddings(model=EMBED_MODEL)


def get_llm():
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def build_vectorstore():
    reset_chroma_dir()

    with open(BASE_DIR / "all_chunks(2025,2026).json", encoding="utf-8") as f:
        all_chunks = json.load(f)

    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name="medical_pdf",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    documents = []
    ids = []

    for i, chunk in enumerate(all_chunks):
        # source_year 표시를 page_content 앞에 추가 → LLM이 년도를 인식할 수 있도록
        source_year = chunk["source"]
        tables_text = "\n".join(chunk["tables"])
        subject = chunk_subject(chunk)
        parent_subject = chunk_parent_subject(chunk)
        page_content = f"[출처년도: {source_year}]\n{subject}\n{chunk['text']}\n{tables_text}".strip()

        doc = Document(
            page_content=page_content,
            metadata={
                "source_year": source_year,
                "parent_subject": parent_subject,
                "subject": subject,
                "section": chunk["section"],
                "page": chunk["page"],
                "text": chunk["text"],
                "tables": json.dumps(chunk["tables"], ensure_ascii=False),
            }
        )
        documents.append(doc)
        ids.append(f"chunk::{i}")

    vectorstore.add_documents(documents=documents, ids=ids)
    print(f"vector db build complete: {len(documents)} chunks")
    return vectorstore


def load_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        collection_name="medical_pdf",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def infer_source_year(question: str) -> str:
    has_2025 = "2025" in question
    has_2026 = "2026" in question

    if has_2025 and has_2026:
        return "2025,2026"
    if has_2025:
        return "2025"
    if has_2026:
        return "2026"
    return ""


def retrieve_docs(vectorstore, question: str, k: int = 4, source_year: str = ""):
    source_year = source_year or infer_source_year(question)
    source_years = [y.strip() for y in source_year.split(",") if y.strip()]

    if len(source_years) == 1:
        return vectorstore.similarity_search(
            question,
            k=k,
            filter={"source_year": source_years[0]},
        )

    if len(source_years) > 1:
        docs = []
        per_year_k = max(1, k // len(source_years))
        for year in source_years:
            docs.extend(
                vectorstore.similarity_search(
                    question,
                    k=per_year_k,
                    filter={"source_year": year},
                )
            )
        return docs

    return vectorstore.similarity_search(question, k=k)


def ask_question(vectorstore, question: str, k: int = 4, source_year: str = ""):
    retrieved_docs = retrieve_docs(vectorstore, question, k=k, source_year=source_year)

    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        source_year = doc.metadata.get("source_year", "?")
        full_context = (
            f"[출처년도: {source_year}]\n"
            f"{doc.page_content}"
        ).strip()
        context_parts.append(f"[문서 {i}] (출처년도={source_year}, page={doc.metadata.get('page')})\n{full_context}")

    context = "\n\n".join(context_parts)

    prompt = f"""아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 [출처년도]가 표시되어 있습니다.
질문이 특정 년도를 묻는 경우 반드시 해당 년도의 출처 컨텍스트만 사용하세요.
컨텍스트에 없는 내용은 "정보를 찾을 수 없습니다"라고 답하세요.
단답으로 대답하세요.(ex. 150,000원, 해당 되지 않음)

[질문]
{question}

[검색 문맥]
{context}""".strip()

    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content


# ── 년도 검색 정확도 판정 ──────────────────────────────────
def check_year_correctness(docs: list, source_year: str) -> bool:
    """
    source_year: "2025", "2026", "2025,2026" 중 하나
    단일 년도: retrieved docs 중 해당 년도 청크가 1개 이상 존재 → True
    cross-year: 2025와 2026 모두 존재 → True
    """
    retrieved_years = {doc.metadata.get("source_year", "") for doc in docs}
    if "," in source_year:
        required = {y.strip() for y in source_year.split(",")}
        return required.issubset(retrieved_years)
    else:
        return source_year.strip() in retrieved_years


def normalize_for_eval(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[,:：;·ㆍ\-\(\)\[\]{}<>→←]', '', text)
    text = text.replace("％", "%")
    return text


def has_no_difference_meaning(text: str) -> bool:
    norm = normalize_for_eval(text)
    return any(
        pattern in norm
        for pattern in [
            "차이없",
            "차이가없",
            "동일",
            "같",
            "변경사항이없",
            "변경사항없",
            "달라지지않",
        ]
    )


def answer_matches_expected(answer: str, expected: str) -> bool:
    answer_norm = normalize_for_eval(answer)
    expected_norm = normalize_for_eval(expected)

    if not expected_norm:
        return False
    return expected_norm in answer_norm


def judge_cross_year_answer(question: str, expected: str, answer: str) -> tuple[bool, str]:
    prompt = f"""당신은 RAG 평가 채점자입니다.
질문, 모범정답, 모델답변을 보고 모델답변이 의미상 정답이면 O, 아니면 X로 판정하세요.

채점 기준:
- 문장부호, 띄어쓰기, 표현 차이는 무시합니다.
- 모범정답의 핵심 연도, 수치, 비교 의미가 모두 맞으면 O입니다.
- 답변이 더 길어도 핵심이 맞으면 O입니다.
- 필요한 연도/수치/조건이 빠졌거나 모범정답과 모순되면 X입니다.
- 출력은 반드시 첫 줄에 `O` 또는 `X`만 쓰고, 둘째 줄에 짧은 이유를 쓰세요.

[질문]
{question}

[모범정답]
{expected}

[모델답변]
{answer}
""".strip()

    response = get_llm().invoke(prompt).content.strip()
    first_line = response.splitlines()[0].strip().upper() if response else ""
    return first_line.startswith("O"), response


# ── 오답 원인 분류 ─────────────────────────────────────────
def classify_cause(answer_ok: bool, search_ok: bool, year_ok: bool,
                   missing_tokens: list) -> tuple[str, str]:
    """
    Returns (verdict, cause)
    verdict: '정답' | '년도오류' | '오답'
    """
    if answer_ok:
        return "정답", ""
    if not search_ok:
        return "오답", f"검색 실패 (누락 토큰: {', '.join(missing_tokens)})"
    if not year_ok:
        return "년도오류", "올바른 주제 검색 성공, 다른 년도 청크 사용"
    return "오답", "LLM 해석 오류 (검색·년도 모두 정상)"


# ── 평가 메인 함수 ─────────────────────────────────────────
def evaluate_golden_dataset(vectorstore, jsonl_path: str):
    rows = []          # 요약 테이블용
    detail_lines = []  # 상세 출력용

    with open(jsonl_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    for item in items:
        q_id        = item["id"]
        question    = item["question"]
        expected    = item["expected_answer"]
        evidence_text = item["evidence_text"]
        difficulty  = item["difficulty"]
        source_year = item.get("source_year", "")

        evidence_tokens = [t.strip().replace(" ", "") for t in evidence_text.split(",")]

        # ── 검색 ──────────────────────────────────────────
        retrieved_docs = retrieve_docs(vectorstore, question, k=4, source_year=source_year)

        full_retrieved_text = "".join(
            doc.page_content + doc.metadata.get("text", "") + doc.metadata.get("tables", "")
            for doc in retrieved_docs
        )

        missing_tokens = [
            t for t in evidence_tokens
            if t.replace(" ", "") not in full_retrieved_text.replace(" ", "")
        ]
        search_ok = len(missing_tokens) == 0
        year_ok   = check_year_correctness(retrieved_docs, source_year)

        # ── 생성 ──────────────────────────────────────────
        answer = ask_question(vectorstore, question, source_year=source_year)
        judge_result = ""
        if difficulty == "cross-year" or "," in source_year:
            answer_ok, judge_result = judge_cross_year_answer(question, expected, answer)
        else:
            answer_ok = answer_matches_expected(answer, expected)

        verdict, cause = classify_cause(answer_ok, search_ok, year_ok, missing_tokens)

        rows.append({
            "q_id":        q_id,
            "difficulty":  difficulty,
            "source_year": source_year,
            "search":      "O" if search_ok else "X",
            "year":        "O" if year_ok   else "X",
            "answer":      answer.replace("\n", " ")[:80],
            "expected":    expected.replace("\n", " ")[:80],
            "verdict":     verdict,
            "cause":       cause,
            "judge":       judge_result.replace("\n", " ")[:120],
        })

        # ── 상세 출력 ──────────────────────────────────────
        retrieved_years_str = ", ".join(
            doc.metadata.get("source_year", "?") for doc in retrieved_docs
        )
        detail_lines.append("=" * 70)
        detail_lines.append(f"질문 ID   : {q_id}  |  난이도: {difficulty}  |  source_year: {source_year}")
        detail_lines.append(f"질문      : {question}")
        detail_lines.append(f"검색 청크 출처년도: [{retrieved_years_str}]")
        detail_lines.append("검색된 청크:")
        for idx, doc in enumerate(retrieved_docs, start=1):
            yr = doc.metadata.get("source_year", "?")
            detail_lines.append(f"\n  [청크 {idx}] (출처년도={yr}, page={doc.metadata.get('page')})")
            detail_lines.append("  " + doc.page_content.replace("\n", "\n  "))
        detail_lines.append(f"\n누락 토큰 : {', '.join(missing_tokens) if missing_tokens else '-'}")
        detail_lines.append(f"검색 성공 : {'O' if search_ok else 'X'}")
        detail_lines.append(f"년도 정확 : {'O' if year_ok else 'X'}")
        detail_lines.append(f"예상 답변 : {expected}")
        detail_lines.append(f"모델 답변 : {answer}")
        if judge_result:
            detail_lines.append(f"LLM 채점  : {judge_result}")
        detail_lines.append(f"판정      : {verdict}  {('← ' + cause) if cause else ''}")
        detail_lines.append("")

    # ── 요약 마크다운 테이블 ──────────────────────────────
    summary_lines = []
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append("## 기록 테이블\n")
    summary_lines.append(
        "| 질문 ID | 난이도 | source_year | 검색된 청크 포함 여부 | 올바른 년도 검색 여부 | LLM 생성 답변 | 정답 | 정답 여부 | 오답 원인 |"
    )
    summary_lines.append(
        "|---------|--------|-------------|---------------------|---------------------|-------------|------|----------|----------|"
    )
    for r in rows:
        summary_lines.append(
            f"| {table_cell(r['q_id'])} | {table_cell(r['difficulty'])} | {table_cell(r['source_year'])} "
            f"| {r['search']} | {r['year']} "
            f"| {table_cell(r['answer'], 60)} | {table_cell(r['expected'], 60)} "
            f"| {table_cell(r['verdict'])} | {table_cell(r['cause'])} |"
        )

    # ── 정답률 집계 ───────────────────────────────────────
    total   = len(rows)
    correct = sum(1 for r in rows if r["verdict"] == "정답")
    year_errors = sum(1 for r in rows if r["verdict"] == "년도오류")
    year_ok_count = sum(1 for r in rows if r["year"] == "O")
    search_ok_count = sum(1 for r in rows if r["search"] == "O")

    summary_lines.append(f"\n| **Basic RAG 정답률** | | | | | | | **{correct}/{total}** | |")

    # ── 난이도별 정답률 ───────────────────────────────────
    difficulties = ["easy", "medium", "hard", "cross-year"]
    summary_lines.append("\n### 난이도별 정답률\n")
    summary_lines.append("| 난이도 | 정답 | 전체 | 정답률 |")
    summary_lines.append("|--------|------|------|--------|")
    for diff in difficulties:
        diff_rows = [r for r in rows if r["difficulty"] == diff]
        if diff_rows:
            d_correct = sum(1 for r in diff_rows if r["verdict"] == "정답")
            summary_lines.append(f"| {diff} | {d_correct} | {len(diff_rows)} | {d_correct/len(diff_rows)*100:.0f}% |")

    # ── 년도 혼동 분석 ────────────────────────────────────
    summary_lines.append("\n### 년도 혼동 분석\n")
    summary_lines.append("| 항목 | 값 |")
    summary_lines.append("|------|-----|")
    summary_lines.append(f"| 검색 성공률 | {search_ok_count}/{total} |")
    summary_lines.append(f"| 올바른 년도 검색 성공률 | {year_ok_count}/{total} |")
    summary_lines.append(f"| 년도 혼동으로 인한 오답 수 | {year_errors} |")
    summary_lines.append(f"| 검색 실패로 인한 오답 수 | {sum(1 for r in rows if '검색 실패' in r['cause'])} |")
    summary_lines.append(f"| LLM 해석 오류로 인한 오답 수 | {sum(1 for r in rows if 'LLM' in r['cause'])} |")

    # 년도 혼동 패턴 (어떤 질문이 년도 오류인지)
    year_error_cases = [r for r in rows if r["verdict"] == "년도오류"]
    if year_error_cases:
        summary_lines.append(f"| 주요 년도 혼동 패턴 | " +
            "; ".join(f"{r['q_id']}({r['source_year']})" for r in year_error_cases) + " |")
    else:
        summary_lines.append("| 주요 년도 혼동 패턴 | 없음 |")

    # ── 최종 출력 합치기 ──────────────────────────────────
    all_output = detail_lines + summary_lines

    output_path = BASE_DIR / "evaluation_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_output))

    print(f"저장 완료: {output_path}")
    print(f"\n[결과 요약]")
    print(f"  정답률 : {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  년도 정확도 : {year_ok_count}/{total} ({year_ok_count/total*100:.1f}%)")
    print(f"  년도 혼동 오답 : {year_errors}건")


def main():
    if REBUILD_DB:
        vectorstore = build_vectorstore()
    else:
        vectorstore = load_vectorstore()

    evaluate_golden_dataset(vectorstore, BASE_DIR / "goldenDataset.jsonl")


if __name__ == "__main__":
    main()
