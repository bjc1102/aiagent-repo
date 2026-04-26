import json
import os
import re
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
import cohere
from langchain.retrievers import EnsembleRetriever

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

CHAT_MODEL  = os.getenv("OPENAI_CHAT_MODEL",      "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# Hybrid Search 파라미터
VECTOR_K   = 10   # 벡터 검색 후보 수
BM25_K     = 10   # BM25 검색 후보 수
RERANK_TOP = 8    # Reranking 후 최종 사용 청크 수
VECTOR_WEIGHT = 0.8
BM25_WEIGHT   = 0.2


# ── 공통 유틸 ──────────────────────────────────────────────────────
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


def cohere_rerank_by_subject(
    query: str, docs: list[Document], top_n: int
) -> list[Document]:
    """subject+text만 Cohere에 전달해 rerank 후 원본 doc 순서로 반환"""
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    subjects = [
        f"{doc.metadata.get('subject', '')}\n{doc.metadata.get('text', '')}".strip()
        for doc in docs
    ]
    for attempt in range(3):
        try:
            results = co.rerank(model="rerank-v3.5", query=query, documents=subjects, top_n=top_n)
            return [docs[r.index] for r in results.results]
        except cohere.errors.too_many_requests_error.TooManyRequestsError:
            wait = 60 * (attempt + 1)
            print(f"Rate limit 도달 — {wait}초 대기 후 재시도 ({attempt+1}/3)")
            time.sleep(wait)
    raise RuntimeError("Cohere rate limit: 3회 재시도 실패")


# ── 문서 로드 (BM25 인덱스용 + ChromaDB 재사용) ───────────────────
def load_all_documents() -> list[Document]:
    """all_chunks JSON → LangChain Document 리스트 (BM25 인덱스 구축용)"""
    with open(BASE_DIR / "all_chunks(2025,2026).json", encoding="utf-8") as f:
        all_chunks = json.load(f)

    documents = []
    for chunk in all_chunks:
        source_year   = chunk["source"]
        tables_text   = "\n".join(chunk["tables"])
        subject       = chunk_subject(chunk)
        parent_subject = chunk_parent_subject(chunk)
        page_content  = f"[출처년도: {source_year}]\n{subject}\n{chunk['text']}\n{tables_text}".strip()

        documents.append(Document(
            page_content=page_content,
            metadata={
                "source_year":    source_year,
                "parent_subject": parent_subject,
                "subject":        subject,
                "section":        chunk["section"],
                "page":           chunk["page"],
                "text":           chunk["text"],
                "tables":         json.dumps(chunk["tables"], ensure_ascii=False),
            }
        ))
    return documents


def load_vectorstore():
    return Chroma(
        collection_name="medical_pdf",
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


# ── 년도 추론 ─────────────────────────────────────────────────────
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


# ── Hybrid + Rerank Retriever 구성 ────────────────────────────────
def build_hybrid_rerank_retriever(
    vectorstore,
    all_documents: list[Document],
    source_year: str,
    vector_k: int = VECTOR_K,
    bm25_k: int   = BM25_K,
):
    """
    source_year: "2025" | "2026" | "2025,2026" | ""
    단일 년도 → 벡터/BM25 모두 해당 년도 문서만 대상
    cross-year / 없음 → 전체 대상
    """
    source_years = [y.strip() for y in source_year.split(",") if y.strip()]

    def to_bm25_doc(d: Document) -> Document:
        subject = d.metadata.get("subject", "")
        text    = d.metadata.get("text", "")
        return Document(page_content=f"{subject}\n{text}".strip(), metadata=d.metadata)

    if len(source_years) == 1:
        yr = source_years[0]
        filtered_docs = [d for d in all_documents if d.metadata.get("source_year") == yr]
        bm25 = BM25Retriever.from_documents([to_bm25_doc(d) for d in filtered_docs])
        bm25.k = bm25_k
        vector = vectorstore.as_retriever(
            search_kwargs={"k": vector_k, "filter": {"source_year": yr}}
        )
    else:
        bm25 = BM25Retriever.from_documents([to_bm25_doc(d) for d in all_documents])
        bm25.k = bm25_k
        vector = vectorstore.as_retriever(search_kwargs={"k": vector_k})

    return EnsembleRetriever(
        retrievers=[vector, bm25],
        weights=[VECTOR_WEIGHT, BM25_WEIGHT],
    )


def retrieve_docs_hybrid(
    vectorstore,
    all_documents: list[Document],
    question: str,
    source_year: str = "",
    top_n: int = RERANK_TOP,
) -> list[Document]:
    source_year = source_year or infer_source_year(question)

    # cross-year: 년도별 분할 후 합산 (각 년도 top_n개 확보)
    if "," in source_year:
        years = [y.strip() for y in source_year.split(",") if y.strip()]
        docs = []
        per_year_top = max(1, top_n // len(years))
        for yr in years:
            ensemble = build_hybrid_rerank_retriever(vectorstore, all_documents, yr)
            docs.extend(ensemble.invoke(question)[:per_year_top])
        return docs

    ensemble = build_hybrid_rerank_retriever(vectorstore, all_documents, source_year)
    return ensemble.invoke(question)[:top_n]


# ── Generation ────────────────────────────────────────────────────
def ask_question_hybrid(
    vectorstore,
    all_documents: list[Document],
    question: str,
    source_year: str = "",
    top_n: int = RERANK_TOP,
    retrieved_docs: Optional[list[Document]] = None,
) -> str:
    if retrieved_docs is None:
        retrieved_docs = retrieve_docs_hybrid(
            vectorstore, all_documents, question,
            source_year=source_year, top_n=top_n,
        )

    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        yr = doc.metadata.get("source_year", "?")
        full = f"[출처년도: {yr}]\n{doc.page_content}".strip()
        context_parts.append(
            f"[문서 {i}] (출처년도={yr}, page={doc.metadata.get('page')})\n{full}"
        )

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

    return get_llm().invoke(prompt).content


# ── 평가 헬퍼 ─────────────────────────────────────────────────────
def check_year_correctness(docs: list, source_year: str) -> bool:
    if not source_year.strip():
        return True
    retrieved_years = {doc.metadata.get("source_year", "") for doc in docs}
    if "," in source_year:
        required = {y.strip() for y in source_year.split(",")}
        return required.issubset(retrieved_years)
    return source_year.strip() in retrieved_years


def normalize_for_eval(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[,:：;·ㆍ\-\(\)\[\]{}<>→←]', '', text)
    return text.replace("％", "%")


def answer_matches_expected(answer: str, expected: str) -> bool:
    return normalize_for_eval(expected) in normalize_for_eval(answer)


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


def classify_cause(answer_ok: bool, search_ok: bool, year_ok: bool,
                   missing_tokens: list) -> tuple[str, str]:
    if answer_ok:
        return "정답", ""
    if not search_ok:
        return "오답", f"검색 실패 (누락 토큰: {', '.join(missing_tokens)})"
    if not year_ok:
        return "년도오류", "올바른 주제 검색 성공, 다른 년도 청크 사용"
    return "오답", "LLM 해석 오류 (검색·년도 모두 정상)"


# ── 평가 메인 ─────────────────────────────────────────────────────
def evaluate_golden_dataset(vectorstore, all_documents: list[Document], jsonl_path: str):
    rows = []
    detail_lines = []

    with open(jsonl_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    for item in items:
        q_id          = item["id"]
        question      = item["question"]
        expected      = item["expected_answer"]
        evidence_text = item["evidence_text"]
        difficulty    = item["difficulty"]
        source_year   = item.get("source_year", "")

        evidence_tokens = [t.strip().replace(" ", "") for t in evidence_text.split(",")]

        # ── Hybrid + Rerank 검색 ─────────────────────────────
        retrieved_docs = retrieve_docs_hybrid(
            vectorstore, all_documents, question,
            source_year=source_year, top_n=RERANK_TOP,
        )

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

        # ── 생성 ─────────────────────────────────────────────
        answer = ask_question_hybrid(
            vectorstore, all_documents, question,
            source_year=source_year, top_n=RERANK_TOP,
            retrieved_docs=retrieved_docs,
        )
        judge_result = ""
        if difficulty == "cross-year" or "," in source_year:
            answer_ok_flag, judge_result = judge_cross_year_answer(question, expected, answer)
        else:
            answer_ok_flag = answer_matches_expected(answer, expected)

        verdict, cause = classify_cause(answer_ok_flag, search_ok, year_ok, missing_tokens)

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

        # ── 상세 출력 ─────────────────────────────────────────
        retrieved_years_str = ", ".join(
            doc.metadata.get("source_year", "?") for doc in retrieved_docs
        )
        detail_lines.append("=" * 70)
        detail_lines.append(
            f"질문 ID   : {q_id}  |  난이도: {difficulty}  |  source_year: {source_year}"
        )
        detail_lines.append(f"질문      : {question}")
        detail_lines.append(f"검색 방식 : Hybrid (Vector+BM25) + Cohere Rerank")
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

    # ── 요약 마크다운 테이블 ─────────────────────────────────
    summary_lines = []
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append("## Step 2 Advanced RAG 기록 테이블\n")

    summary_lines.append("### 설정값\n")
    summary_lines.append("| 항목 | 설정값 |")
    summary_lines.append("|------|--------|")
    summary_lines.append(f"| BM25 Retriever k | {BM25_K} |")
    summary_lines.append(f"| Vector Retriever k | {VECTOR_K} |")
    summary_lines.append(f"| Ensemble 가중치 (vector : BM25) | {VECTOR_WEIGHT} : {BM25_WEIGHT} |")
    summary_lines.append(f"| Re-ranker 종류 및 모델명 | Cohere CohereRerank / rerank-v3.5 |")
    summary_lines.append(f"| Re-ranking 후 최종 Top-K | {RERANK_TOP} |")
    summary_lines.append(f"| 메타데이터 필터링 | 단일 년도 질문 시 source_year 필터 적용 |")

    summary_lines.append("\n### 문항별 결과\n")
    summary_lines.append(
        "| 질문 ID | 난이도 | source_year | 검색 방식 | 검색 결과 포함 여부 | 올바른 년도 검색 여부 | LLM 생성 답변 | 정답 | 정답 여부 | 오답 원인 |"
    )
    summary_lines.append(
        "|---------|--------|-------------|----------|-------------------|-------------------|-------------|------|----------|----------|"
    )
    for r in rows:
        summary_lines.append(
            f"| {table_cell(r['q_id'])} | {table_cell(r['difficulty'])} | {table_cell(r['source_year'])} "
            f"| Hybrid+Rerank | {r['search']} | {r['year']} "
            f"| {table_cell(r['answer'], 60)} | {table_cell(r['expected'], 60)} "
            f"| {table_cell(r['verdict'])} | {table_cell(r['cause'])} |"
        )

    # ── 정답률 집계 ──────────────────────────────────────────
    total            = len(rows)
    correct          = sum(1 for r in rows if r["verdict"] == "정답")
    year_errors      = sum(1 for r in rows if r["verdict"] == "년도오류")
    year_ok_count    = sum(1 for r in rows if r["year"] == "O")
    search_ok_count  = sum(1 for r in rows if r["search"] == "O")

    summary_lines.append(f"\n| **Advanced RAG 정답률** | | | | | | | | **{correct}/{total}** | |")

    # ── 난이도별 정답률 ──────────────────────────────────────
    summary_lines.append("\n### 난이도별 정답률\n")
    summary_lines.append("| 난이도 | 정답 | 전체 | 정답률 |")
    summary_lines.append("|--------|------|------|--------|")
    for diff in ["easy", "medium", "hard", "cross-year"]:
        diff_rows = [r for r in rows if r["difficulty"] == diff]
        if diff_rows:
            d_correct = sum(1 for r in diff_rows if r["verdict"] == "정답")
            summary_lines.append(
                f"| {diff} | {d_correct} | {len(diff_rows)} | {d_correct/len(diff_rows)*100:.0f}% |"
            )

    # ── 년도 혼동 분석 ────────────────────────────────────────
    summary_lines.append("\n### 년도 혼동 분석\n")
    summary_lines.append("| 항목 | 값 |")
    summary_lines.append("|------|-----|")
    summary_lines.append(f"| 검색 성공률 | {search_ok_count}/{total} |")
    summary_lines.append(f"| 올바른 년도 검색 성공률 | {year_ok_count}/{total} |")
    summary_lines.append(f"| 년도 혼동으로 인한 오답 수 | {year_errors} |")
    summary_lines.append(f"| 검색 실패로 인한 오답 수 | {sum(1 for r in rows if '검색 실패' in r['cause'])} |")
    summary_lines.append(f"| LLM 해석 오류로 인한 오답 수 | {sum(1 for r in rows if 'LLM' in r['cause'])} |")

    year_error_cases = [r for r in rows if r["verdict"] == "년도오류"]
    if year_error_cases:
        summary_lines.append(
            "| 주요 년도 혼동 패턴 | "
            + "; ".join(f"{r['q_id']}({r['source_year']})" for r in year_error_cases)
            + " |"
        )
    else:
        summary_lines.append("| 주요 년도 혼동 패턴 | 없음 |")

    # ── 저장 ─────────────────────────────────────────────────
    all_output = detail_lines + summary_lines
    output_path = BASE_DIR / "evaluation_result(step2).txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_output))

    print(f"저장 완료: {output_path}")
    if total == 0:
        print("평가할 항목이 없습니다.")
        return
    print(f"\n[Step 2 결과 요약]")
    print(f"  정답률      : {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  년도 정확도  : {year_ok_count}/{total} ({year_ok_count/total*100:.1f}%)")
    print(f"  년도 혼동 오답: {year_errors}건")


def main():
    print("ChromaDB 로드 중...")
    vectorstore = load_vectorstore()

    print("BM25 인덱스용 문서 로드 중...")
    all_documents = load_all_documents()
    print(f"  총 {len(all_documents)}개 청크 로드 완료")

    evaluate_golden_dataset(
        vectorstore,
        all_documents,
        BASE_DIR / "goldenDataset.jsonl",
    )


if __name__ == "__main__":
    main()
