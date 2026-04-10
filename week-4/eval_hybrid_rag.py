from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from rag_pipeline import (
    resolve_runtime_config,
    load_vectorstore,
    search_documents_hybrid,
    generate_answer,
    extract_year_from_question,
    INDEX_DIR,
)


BASE_DIR = Path(__file__).resolve().parent

CANDIDATE_DATASET_PATHS = [
    BASE_DIR / "data" / "golden_dataset.jsonl",
    BASE_DIR / "golden_dataset.jsonl",
]

DATASET_PATH = None
for p in CANDIDATE_DATASET_PATHS:
    if p.exists():
        DATASET_PATH = p
        break

if DATASET_PATH is None:
    DATASET_PATH = CANDIDATE_DATASET_PATHS[0]

OUTPUT_DIR = BASE_DIR / "outputs"
RESULT_PATH = OUTPUT_DIR / "hybrid_eval_results.json"
JSONL_RESULT_PATH = OUTPUT_DIR / "hybrid_eval_results.jsonl"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"golden dataset 파일을 찾을 수 없습니다: {path}")

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(",", "")
    s = s.replace(" %", "%").replace("% ", "%")
    s = s.replace(" 원", "원")
    s = s.replace("입니다.", "이다")
    s = s.replace("입니다", "이다")
    s = s.replace("까지이다.", "까지")
    s = s.replace("까지이다", "까지")
    s = s.rstrip(".")
    return s


def extract_core_tokens(text: str) -> List[str]:
    s = normalize_text(text)
    tokens: List[str] = []

    patterns = [
        r"\d+원",
        r"\d+%",
        r"\d+일",
        r"\d+회",
        r"20\d{2}년\s*\d{1,2}월\s*\d{1,2}일",
    ]

    for pattern in patterns:
        tokens.extend(re.findall(pattern, s))

    keyword_candidates = [
        "무료", "면제", "가능", "불가능", "필요하지 않다", "필요하다",
        "우울증", "조기정신증", "이상지질혈증",
        "의원", "보건소", "보건지소", "보건진료소", "보건의료원", "약국",
        "예약접수일", "제출일", "의료급여의뢰서", "전액", "본인부담",
        "장기지속형 주사제", "선택의료급여기관", "등록 장애인", "15세 이하 아동",
        "제2차의료급여기관", "제3차의료급여기관"
    ]

    for kw in keyword_candidates:
        if kw in s:
            tokens.append(kw)

    return sorted(set(tokens), key=lambda x: (-len(x), x))


def get_strict_tokens(tokens: List[str]) -> List[str]:
    return [
        tok for tok in tokens
        if re.search(r"\d+원|\d+%|\d+일|\d+회|20\d{2}년\s*\d{1,2}월\s*\d{1,2}일", tok)
    ]


def judge_answer(pred_answer: str, expected_answer: str) -> bool:
    pred = normalize_text(pred_answer)
    gold = normalize_text(expected_answer)

    if not pred or "[llm generation failed]" in pred:
        return False

    if gold in pred:
        return True

    gold_tokens = extract_core_tokens(expected_answer)
    if not gold_tokens:
        return False

    strict_tokens = get_strict_tokens(gold_tokens)
    if strict_tokens and not all(tok in pred for tok in strict_tokens):
        return False

    matched = sum(1 for tok in gold_tokens if tok in pred)

    if len(gold_tokens) == 1:
        return matched == 1

    return matched >= max(1, len(gold_tokens) // 2)


def judge_year_correct(question: str, dataset_source_year: str, retrieved_docs: List[Any]) -> bool:
    question_year = extract_year_from_question(question) or str(dataset_source_year)
    retrieved_years = [str(doc.metadata.get("source_year", "unknown")) for doc in retrieved_docs]
    return question_year in retrieved_years if retrieved_years else False


def judge_chunk_hit(expected_answer: str, retrieved_docs: List[Any]) -> bool:
    gold_tokens = extract_core_tokens(expected_answer)
    if not gold_tokens:
        return False

    strict_tokens = get_strict_tokens(gold_tokens)

    for doc in retrieved_docs:
        content = normalize_text(doc.page_content)

        if strict_tokens and all(tok in content for tok in strict_tokens):
            return True

        matched = sum(1 for tok in gold_tokens if tok in content)
        if matched >= max(1, len(gold_tokens) // 2):
            return True

    return False


def infer_error_reason(is_correct: bool, chunk_hit: bool, year_correct: bool, generated_answer: str) -> str:
    if is_correct:
        return ""

    gen = normalize_text(generated_answer)

    if "[llm generation failed]" in gen:
        return "llm_generation_failed"
    if not year_correct:
        return "year_confusion"
    if year_correct and not chunk_hit:
        return "retrieval_miss"
    if year_correct and chunk_hit:
        return "generation_miss"
    return "unknown"


def doc_to_brief(doc: Any) -> Dict[str, Any]:
    return {
        "source_year": doc.metadata.get("source_year"),
        "source_file": doc.metadata.get("source_file"),
        "page": doc.metadata.get("page"),
        "preview": doc.page_content[:250].replace("\n", " "),
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    print(f"DATASET_PATH: {DATASET_PATH}")
    print(f"INDEX_DIR   : {INDEX_DIR}")

    dataset_rows = load_jsonl(DATASET_PATH)
    print(f"Loaded dataset rows: {len(dataset_rows)}")

    resolve_runtime_config()
    vectorstore = load_vectorstore(INDEX_DIR)

    results: List[Dict[str, Any]] = []

    total = 0
    answer_correct_count = 0
    year_correct_count = 0
    chunk_hit_count = 0

    for idx, row in enumerate(dataset_rows, start=1):
        question = row["question"]
        expected_answer = row["expected_answer"]
        source_year = str(row["source_year"])
        difficulty = row.get("difficulty", "")
        question_id = row.get("id", f"q{idx:02d}")

        retrieved_docs = search_documents_hybrid(
            vectorstore=vectorstore,
            question=question,
            top_k=4,
            use_year_filter=True,
        )

        generated_answer = generate_answer(question=question, docs=retrieved_docs)

        chunk_hit = judge_chunk_hit(expected_answer, retrieved_docs)
        year_correct = judge_year_correct(question, source_year, retrieved_docs)
        is_correct = judge_answer(generated_answer, expected_answer)
        error_reason = infer_error_reason(
            is_correct=is_correct,
            chunk_hit=chunk_hit,
            year_correct=year_correct,
            generated_answer=generated_answer,
        )

        total += 1
        answer_correct_count += int(is_correct)
        year_correct_count += int(year_correct)
        chunk_hit_count += int(chunk_hit)

        result_row = {
            "question_id": question_id,
            "difficulty": difficulty,
            "source_year": source_year,
            "question": question,
            "expected_answer": expected_answer,
            "retrieved_chunk_hit": "O" if chunk_hit else "X",
            "year_correct": "O" if year_correct else "X",
            "generated_answer": generated_answer,
            "is_correct": "O" if is_correct else "X",
            "error_reason": error_reason,
            "retrieved_docs": [doc_to_brief(doc) for doc in retrieved_docs],
        }
        results.append(result_row)

        print(
            f"[{question_id}] "
            f"difficulty={difficulty} | "
            f"year={source_year} | "
            f"chunk_hit={'O' if chunk_hit else 'X'} | "
            f"year_correct={'O' if year_correct else 'X'} | "
            f"correct={'O' if is_correct else 'X'} | "
            f"error={error_reason or '-'}"
        )

    summary = {
        "step": "hybrid_rag",
        "total": total,
        "answer_accuracy": round(answer_correct_count / total, 4) if total else 0.0,
        "year_accuracy": round(year_correct_count / total, 4) if total else 0.0,
        "chunk_hit_rate": round(chunk_hit_count / total, 4) if total else 0.0,
        "answer_correct_count": answer_correct_count,
        "year_correct_count": year_correct_count,
        "chunk_hit_count": chunk_hit_count,
        "results": results,
    }

    save_json(RESULT_PATH, summary)
    save_jsonl(JSONL_RESULT_PATH, results)

    print("\n=== Hybrid RAG Summary ===")
    print(f"Total           : {total}")
    print(f"Answer Accuracy : {summary['answer_accuracy'] * 100:.2f}%")
    print(f"Year Accuracy   : {summary['year_accuracy'] * 100:.2f}%")
    print(f"Chunk Hit Rate  : {summary['chunk_hit_rate'] * 100:.2f}%")
    print(f"Saved JSON      : {RESULT_PATH}")
    print(f"Saved JSONL     : {JSONL_RESULT_PATH}")


if __name__ == "__main__":
    main()