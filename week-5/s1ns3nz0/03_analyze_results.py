"""
Step 3 — Read CSVs from Step 2 and produce:
  - Markdown tables (전체 평균, 문항별, 4주차 vs Ragas)
  - Failure case selection (Case A / B / C)

OUTPUTS:
  - analysis_tables.md          — paste-ready markdown
  - failure_cases.json          — selected qids per case
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
B_CSV = HERE / "basic_ragas_scores.csv"
A_CSV = HERE / "advanced_ragas_scores.csv"
W4_BASIC = HERE.parent.parent / "week-4" / "s1ns3nz0" / "basic_rag_results.json"
W4_ADV   = HERE.parent.parent / "week-4" / "s1ns3nz0" / "advanced_rag_results.json"


METRIC_COLS = [
    "context_recall",
    "llm_context_precision_with_reference",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]
METRIC_LABELS = {
    "context_recall": "Context Recall",
    "llm_context_precision_with_reference": "Context Precision",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "answer_correctness": "Answer Correctness",
}


def safe_mean(s):
    try:
        return float(pd.to_numeric(s, errors="coerce").mean())
    except Exception:
        return float("nan")


def fmt(x):
    return "—" if pd.isna(x) else f"{x:.3f}"


def overall_table(b: pd.DataFrame, a: pd.DataFrame) -> str:
    rows = ["| 메트릭 | Basic | Advanced | Δ (A−B) |", "|--------|-------|----------|---------|"]
    for c in METRIC_COLS:
        if c not in b.columns:
            continue
        bm, am = safe_mean(b[c]), safe_mean(a[c])
        delta = am - bm
        rows.append(f"| {METRIC_LABELS[c]} | {fmt(bm)} | {fmt(am)} | {fmt(delta)} |")
    return "\n".join(rows)


def per_question_table(b: pd.DataFrame, a: pd.DataFrame) -> str:
    """
    | qid | difficulty | source_year | Ctx R (B/A) | Ctx P (B/A) | Faith (B/A) | Ans R (B/A) | Ans C (B/A) |
    """
    cols = ["qid", "difficulty", "source_year"] + [f"{METRIC_LABELS[c]} (B/A)" for c in METRIC_COLS]
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    n = len(b)
    for i in range(n):
        rb, ra = b.iloc[i], a.iloc[i]
        line = [rb["id"], rb["difficulty"], rb["source_year"]]
        for c in METRIC_COLS:
            if c not in b.columns:
                line.append("—")
                continue
            bv = pd.to_numeric(rb.get(c), errors="coerce")
            av = pd.to_numeric(ra.get(c), errors="coerce")
            line.append(f"{fmt(bv)} / {fmt(av)}")
        rows.append("| " + " | ".join(line) + " |")
    return "\n".join(rows)


def vs_week4_table(b: pd.DataFrame, a: pd.DataFrame) -> str:
    """Ragas q01..qN matches week-4 q01..q24 by **question text**, not index."""
    if not (W4_BASIC.exists() and W4_ADV.exists()):
        return "_(week-4 results files not found — skipping comparison)_"

    w4b = {r["question"]: r for r in json.loads(W4_BASIC.read_text())["results"]}
    w4a = {r["question"]: r for r in json.loads(W4_ADV.read_text())["results"]}

    rows = ["| qid | 질문 | 4주차 Basic 판정 | Ragas Ans Corr (B) | 4주차 Adv 판정 | Ragas Ans Corr (A) | 일치(B) | 일치(A) |",
            "|---|---|---|---|---|---|---|---|"]
    for i in range(len(b)):
        rb, ra = b.iloc[i], a.iloc[i]
        q = rb["user_input"]
        bv = pd.to_numeric(rb.get("answer_correctness"), errors="coerce")
        av = pd.to_numeric(ra.get("answer_correctness"), errors="coerce")
        bj = w4b.get(q, {}).get("is_correct")
        aj = w4a.get(q, {}).get("is_correct")
        # Threshold for "correct" in Ragas: ≥ 0.7 by convention
        bc = (bv >= 0.7) if not pd.isna(bv) else None
        ac = (av >= 0.7) if not pd.isna(av) else None
        match_b = "✓" if (bc is not None and bj is not None and bool(bc) == bool(bj)) else "✗"
        match_a = "✓" if (ac is not None and aj is not None and bool(ac) == bool(aj)) else "✗"
        rows.append(
            f"| {rb['id']} | {q[:40]}… | "
            f"{'정답' if bj else '오답' if bj is False else '—'} | {fmt(bv)} | "
            f"{'정답' if aj else '오답' if aj is False else '—'} | {fmt(av)} | "
            f"{match_b} | {match_a} |"
        )
    return "\n".join(rows)


def select_failure_cases(b: pd.DataFrame, a: pd.DataFrame) -> dict:
    """
    Case A: Advanced < Basic by ≥ 0.15 on Answer Correctness  (regression)
    Case B: source_year = "2025+2026" with Faithfulness ≥ 0.7 but Ans Corr < 0.5
    Case C: Faith ≥ 0.85 but Ans Corr ≤ 0.55  on either pipeline
    """
    cases = {"A": [], "B": [], "C": []}
    for i in range(len(b)):
        rb, ra = b.iloc[i], a.iloc[i]
        bv = pd.to_numeric(rb.get("answer_correctness"), errors="coerce")
        av = pd.to_numeric(ra.get("answer_correctness"), errors="coerce")
        bf = pd.to_numeric(rb.get("faithfulness"), errors="coerce")
        af = pd.to_numeric(ra.get("faithfulness"), errors="coerce")

        # Case A
        if not pd.isna(bv) and not pd.isna(av) and bv - av >= 0.15:
            cases["A"].append({"id": rb["id"], "delta": float(bv - av),
                               "basic_ac": float(bv), "advanced_ac": float(av)})
        # Case B (year confusion potential)
        if rb["source_year"] in ("2025+2026", "cross-year"):
            if not pd.isna(af) and not pd.isna(av) and af >= 0.7 and av < 0.5:
                cases["B"].append({"id": rb["id"], "advanced_faith": float(af),
                                   "advanced_ac": float(av)})
        # Case C
        for label, fv, cv in [("basic", bf, bv), ("advanced", af, av)]:
            if not pd.isna(fv) and not pd.isna(cv) and fv >= 0.85 and cv <= 0.55:
                cases["C"].append({"id": rb["id"], "pipeline": label,
                                   "faith": float(fv), "ans_corr": float(cv)})
    return cases


def main():
    if not (B_CSV.exists() and A_CSV.exists()):
        print("CSV files missing. Run 02_ragas_evaluate.py first.")
        return
    b = pd.read_csv(B_CSV)
    a = pd.read_csv(A_CSV)

    out = ["# Step 3 — Aggregated Tables\n",
           "## 4-2. 5메트릭 결과 (전체 평균)", "", overall_table(b, a), "",
           "## 4-3. 문항별 메트릭 (B/A)", "", per_question_table(b, a), "",
           "## 4-4. 4주차 수동 채점 vs 5주차 Ragas Answer Correctness",
           "(Ragas Ans Corr ≥ 0.7 → \"정답\"으로 판정해 비교)", "",
           vs_week4_table(b, a)]
    md = "\n".join(out)
    (HERE / "analysis_tables.md").write_text(md, encoding="utf-8")
    print(md[:2000])
    print("\n…\n  saved →", HERE / "analysis_tables.md")

    cases = select_failure_cases(b, a)
    (HERE / "failure_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Failure cases:", json.dumps(cases, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
