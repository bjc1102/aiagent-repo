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

    with open(BASE_DIR / "data" / "chunks" / "all_chunks(2025,2026).json", encoding="utf-8") as f:
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


def format_retrieved_contexts(retrieved_docs: list[Document]) -> tuple[list[str], str]:
    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        source_year = doc.metadata.get("source_year", "?")
        full_context = (
            f"[출처년도: {source_year}]\n"
            f"{doc.page_content}"
        ).strip()
        context_parts.append(
            f"[문서 {i}] (출처년도={source_year}, page={doc.metadata.get('page')})\n{full_context}"
        )

    return context_parts, "\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> str:
    return f"""당신은 의료급여 문서를 바탕으로 답하는 RAG assistant입니다.
아래 검색 문맥만 사용해 질문에 답하세요.

답변 규칙:
1. 질문이 특정 년도를 묻는 경우 반드시 해당 년도의 출처 문맥만 사용하세요.
2. 답변에는 질문의 핵심 조건과 결론을 함께 적으세요.
3. 숫자나 퍼센트만 단독으로 답하지 마세요.
4. 반드시 완전한 문장으로 답하고, 년도·대상·항목·수치가 모두 드러나게 한 문장으로만 답하세요.
5. 두 개 이상의 년도를 비교하는 질문이면 각 년도 값을 분리해서 분명하게 답하세요.
6. 문맥에 없는 내용은 추측하지 말고 "정보를 찾을 수 없습니다."라고 답하세요.
7. 이유 설명, 계산 과정 설명, 배경 설명, 추가 해설은 쓰지 마세요.
8. 아래처럼 값만 쓰는 답변은 금지입니다: `15%`, `40,000원`, `해당 없음`
9. 답변은 정답 문장 하나로 끝내세요. `이는`, `따라서`, `왜냐하면`, `계산하면`, `감소합니다` 같은 설명형 표현은 쓰지 마세요.

출력 형식:
- 단일 값 질문: `2026년 의료급여 65세 이상 2종 수급권자의 틀니 본인부담률은 15%입니다.`
- 계산 질문: `2025년 2종 수급권자가 협착증으로 복잡추나 치료를 받고 비용이 100,000원인 경우 본인부담금은 40,000원입니다.`
- 비교 질문: `항정신병 장기지속형 주사제 본인부담률은 2025년에는 5%, 2026년에는 2%입니다.`

좋은 답변 예시:
- 2026년 의료급여 65세 이상 2종 수급권자의 틀니 본인부담률은 15%입니다.
- 항정신병 장기지속형 주사제 본인부담률은 2025년에는 5%, 2026년에는 2%입니다.

나쁜 답변 예시:
- 15%
- 40,000원
- 2025년 2종 수급권자의 본인부담금은 40,000원입니다. 이는 본인부담률이 40%이기 때문입니다.

[질문]
{question}

[검색 문맥]
{context}""".strip()


def generate_answer_from_docs(question: str, retrieved_docs: list[Document]) -> str:
    _, context = format_retrieved_contexts(retrieved_docs)
    llm = get_llm()
    response = llm.invoke(build_prompt(question, context))
    return str(response.content)


def run_pipeline(vectorstore, question: str, k: int = 4, source_year: str = "") -> dict:
    retrieved_docs = retrieve_docs(vectorstore, question, k=k, source_year=source_year)
    retrieved_contexts, _ = format_retrieved_contexts(retrieved_docs)
    response = generate_answer_from_docs(question, retrieved_docs)

    return {
        "question": question,
        "source_year": source_year or infer_source_year(question),
        "retrieved_docs": retrieved_docs,
        "retrieved_contexts": retrieved_contexts,
        "response": response,
    }


def ask_question(vectorstore, question: str, k: int = 4, source_year: str = ""):
    return run_pipeline(
        vectorstore,
        question,
        k=k,
        source_year=source_year,
    )["response"]


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


def answer_matches_expected(answer: str, expected: str) -> bool:
    answer_norm = normalize_for_eval(answer)
    expected_norm = normalize_for_eval(expected)

    if not expected_norm:
        return False
    return expected_norm in answer_norm


def judge_cross_year_answer(question: str, expected: str, answer: str, context: str) -> tuple[bool, str]:
    prompt = f"""당신은 RAG 답변 평가자입니다.
질문, 검색 문맥, 모범정답, 모델답변을 읽고 Ragas 관점의 4개 항목으로만 평가하세요.

[Task Introduction]
- 의료급여 Q&A에 대한 RAG 답변 품질을 평가합니다.

[Evaluation Criteria]
1. faithfulness (1~5): 모델답변이 검색 문맥에 얼마나 충실한가
   - 5: 핵심 주장과 수치가 모두 검색 문맥으로 직접 뒷받침됨
   - 4: 대부분 뒷받침되며 검색 문맥의 수치/조건만으로 가능한 단순 계산 또는 비교만 추가됨
   - 3: 일부만 문맥으로 뒷받침됨
   - 2: 중요한 부분이 문맥으로 충분히 뒷받침되지 않음
   - 1: 핵심 답변이 문맥과 맞지 않거나 문맥 밖 정보에 의존함
2. answer_relevancy (1~5): 모델답변이 질문 의도에 얼마나 직접 답하는가
   - 5: 질문이 묻는 핵심을 직접 답함
   - 4: 대체로 직접 답하지만 약간의 군더더기가 있음
   - 3: 질문 일부만 답함
   - 2: 질문 초점을 많이 벗어남
   - 1: 질문과 거의 관련 없음
3. context_recall (1~5): 검색 문맥에 답에 필요한 정보가 얼마나 충분히 포함되어 있는가
   - 5: 답에 필요한 핵심 년도, 대상, 조건, 수치가 모두 검색 문맥에 있음
   - 4: 거의 다 있으나 사소한 정보가 부족함
   - 3: 핵심 정보 일부가 빠져 있음
   - 2: 중요한 정보가 여러 개 빠져 있음
   - 1: 답에 필요한 정보가 대부분 없음
4. context_precision (1~5): 검색 문맥이 질문과 얼마나 관련성 높게 정리되어 있는가
   - 5: 상위 검색 문맥이 대부분 질문과 직접 관련 있음
   - 4: 대체로 관련 있으나 일부 불필요한 문맥이 섞임
   - 3: 관련 문맥과 불필요한 문맥이 섞여 있음
   - 2: 불필요한 문맥 비중이 큼
   - 1: 검색 문맥 대부분이 질문과 관련 없음

[Evaluation Steps]
1. 질문이 요구하는 핵심 요소(년도, 대상, 항목, 조건, 수치)를 먼저 정리합니다.
2. 모델답변의 핵심 주장들이 검색 문맥으로 뒷받침되는지 faithfulness를 평가합니다.
   - 답변의 핵심 사실이 검색 문맥에 직접 있거나,
   - 검색 문맥의 수치/조건만으로 가능한 단순 계산(비율 계산, 금액 계산, 연도별 비교)이라면 grounded로 인정합니다.
   - 다만 검색 문맥에 없는 새로운 사실, 조건, 예외, 해석을 추가하면 grounded가 아닙니다.
3. 모델답변이 질문 의도에 직접 답하는지 answer_relevancy를 평가합니다.
4. 검색 문맥에 답에 필요한 정보가 충분한지 context_recall을 평가합니다.
5. 검색 문맥의 관련성 밀도가 높은지 context_precision을 평가합니다.
6. verdict는 핵심 년도/조건/수치가 맞고 문맥과 모순되지 않을 때만 O, 아니면 X로 줍니다.

[Example Output]
verdict: O
faithfulness: 5
answer_relevancy: 5
context_recall: 5
context_precision: 4
rationale: 핵심 수치와 조건이 모두 맞고 검색 문맥으로 뒷받침됩니다.

[Evaluation Form]
반드시 아래 형식 그대로 출력하세요.
verdict: [O 또는 X]
faithfulness: [1~5 정수]
answer_relevancy: [1~5 정수]
context_recall: [1~5 정수]
context_precision: [1~5 정수]
rationale: [3문장 이내]

[질문]
{question}

[검색 문맥]
{context}

[모범정답]
{expected}

[모델답변]
{answer}
""".strip()

    content = get_llm().invoke(prompt).content
    if isinstance(content, list):
        response_text = str(content[0].get("content", "") if content and isinstance(content[0], dict) else content[0] if content else "")
    else:
        response_text = str(content)
    response_text = response_text.strip()
    verdict_match = re.search(r"verdict\s*:\s*([OX])", response_text, re.IGNORECASE)
    if verdict_match:
        return verdict_match.group(1).upper() == "O", response_text
    first_line = response_text.splitlines()[0].strip().upper() if response_text else ""
    return first_line.startswith("O"), response_text

# ── 평가 메인 함수 ─────────────────────────────────────────
def evaluate_golden_dataset(vectorstore, jsonl_path: str):
    rows = []          # 요약 테이블용
    detail_lines = []  # 상세 출력용

    with open(jsonl_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    for item in items:
        q_id        = item["id"]
        question    = item["question"]
        expected    = item["ground_truth"]
        difficulty  = item["difficulty"]
        source_year = item.get("source_year", "")

        # ── 검색 ──────────────────────────────────────────
        retrieved_docs = retrieve_docs(vectorstore, question, k=4, source_year=source_year)
        _, context_text = format_retrieved_contexts(retrieved_docs)
        search_ok = bool(retrieved_docs)
        year_ok   = check_year_correctness(retrieved_docs, source_year)

        # ── 생성 ──────────────────────────────────────────
        answer = generate_answer_from_docs(question, retrieved_docs)
        answer_ok, judge_result = judge_cross_year_answer(question, expected, answer, context_text)
        verdict = "정답" if answer_ok else "오답"

        rows.append({
            "q_id":        q_id,
            "difficulty":  difficulty,
            "source_year": source_year,
            "search":      "O" if search_ok else "X",
            "year":        "O" if year_ok   else "X",
            "answer":      answer.replace("\n", " ")[:80],
            "expected":    expected.replace("\n", " ")[:80],
            "verdict":     verdict,
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
        detail_lines.append(f"검색 성공 : {'O' if search_ok else 'X'}")
        detail_lines.append(f"년도 정확 : {'O' if year_ok else 'X'}")
        detail_lines.append(f"예상 답변 : {expected}")
        detail_lines.append(f"모델 답변 : {answer}")
        if judge_result:
            detail_lines.append(f"LLM 채점  : {judge_result}")
        detail_lines.append(f"판정      : {verdict}")
        detail_lines.append("")

    # ── 요약 마크다운 테이블 ──────────────────────────────
    summary_lines = []
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append("## 기록 테이블\n")
    summary_lines.append(
        "| 질문 ID | 난이도 | source_year | 검색 결과 존재 여부 | 올바른 년도 검색 여부 | LLM 생성 답변 | 정답 | 정답 여부 |"
    )
    summary_lines.append(
        "|---------|--------|-------------|---------------------|---------------------|-------------|------|----------|"
    )
    for r in rows:
        summary_lines.append(
            f"| {table_cell(r['q_id'])} | {table_cell(r['difficulty'])} | {table_cell(r['source_year'])} "
            f"| {r['search']} | {r['year']} "
            f"| {table_cell(r['answer'], 60)} | {table_cell(r['expected'], 60)} "
            f"| {table_cell(r['verdict'])} |"
        )

    # ── 정답률 집계 ───────────────────────────────────────
    total   = len(rows)
    correct = sum(1 for r in rows if r["verdict"] == "정답")
    year_ok_count = sum(1 for r in rows if r["year"] == "O")
    search_ok_count = sum(1 for r in rows if r["search"] == "O")

    summary_lines.append(f"\n| **Basic RAG 정답률** | | | | | | | **{correct}/{total}** |")

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

    # ── 검색/년도 진단 ────────────────────────────────────
    summary_lines.append("\n### 검색/년도 진단\n")
    summary_lines.append("| 항목 | 값 |")
    summary_lines.append("|------|-----|")
    summary_lines.append(f"| 검색 성공률 | {search_ok_count}/{total} |")
    summary_lines.append(f"| 올바른 년도 검색 성공률 | {year_ok_count}/{total} |")

    # ── 최종 출력 합치기 ──────────────────────────────────
    all_output = detail_lines + summary_lines

    output_path = BASE_DIR / "evaluation_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_output))

    print(f"저장 완료: {output_path}")
    print(f"\n[결과 요약]")
    print(f"  정답률 : {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  년도 정확도 : {year_ok_count}/{total} ({year_ok_count/total*100:.1f}%)")


def main():
    if REBUILD_DB:
        vectorstore = build_vectorstore()
    else:
        vectorstore = load_vectorstore()

    evaluate_golden_dataset(vectorstore, str(BASE_DIR / "goldenDataset.jsonl"))


if __name__ == "__main__":
    main()
