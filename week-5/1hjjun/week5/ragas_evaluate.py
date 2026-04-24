"""5주차 Ragas 자동 평가 스크립트.

입력:
  - ../golden_dataset_v2.jsonl (ground_truth, ground_truth_contexts 포함)
  - ../week3/RAGresult.json  (Basic RAG 실행 결과)
  - ../week4/advanced_result.json (Advanced RAG 실행 결과)

판정자(Judge LLM): Gemini 2.5 Flash (무료 티어 사용 가능, 빠른 응답)
  ※ 원래 Claude Sonnet 4.5 → GPT-4o → Gemini 2.5 Pro 순 시도했으나 모두 비용·타임아웃 이슈.
임베딩: Google gemini-embedding-001 (4주차 재사용)

샘플 제한: 환경변수 RAGAS_LIMIT 로 첫 N문항만 평가 (비용 절감).
  예) RAGAS_LIMIT=5 python3 ragas_evaluate.py

메트릭 (5개):
  1) LLMContextRecall  — Retrieval 재현율
  2) LLMContextPrecisionWithReference — Retrieval 정밀도
  3) Faithfulness — Generation 환각 체크
  4) ResponseRelevancy — Generation 질문 관련성
  5) AnswerCorrectness — End-to-end 정답 일치

출력:
  - basic_ragas_scores.csv
  - advanced_ragas_scores.csv
  - ragas_summary.json (평균 점수 요약)
"""

import os
import json
import sys
import pandas as pd
from dotenv import load_dotenv

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    LLMContextRecall,
    LLMContextPrecisionWithReference,
    Faithfulness,
    ResponseRelevancy,
    AnswerCorrectness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Judge LLM: Gemini 2.5 Flash
# =============================================================================
def build_judge_llm():
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    if not gemini_key or gemini_key == "your_api_key_here":
        sys.exit("❌ GEMINI_API_KEY 필요 (.env 확인)")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        timeout=120,
        max_retries=2,
    )


# =============================================================================
# 데이터 로드
# =============================================================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_eval_dataset(golden, rag_result):
    """golden_dataset_v2 + RAG 결과 → SingleTurnSample 리스트."""
    golden_by_id = {g["id"]: g for g in golden}
    samples = []
    for r in rag_result:
        q_id = r["id"]
        g = golden_by_id.get(q_id)
        if not g:
            print(f"⚠️ golden dataset에 {q_id} 없음 — 건너뜀")
            continue
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["generated_answer"],
            retrieved_contexts=r.get("retrieved_contexts", []),
            reference=g["ground_truth"],
            reference_contexts=g["ground_truth_contexts"],
        )
        samples.append(sample)
    return EvaluationDataset(samples=samples)


# =============================================================================
# 평가 실행 — Two-pass (Gemini Pro → GPT-4o)
# =============================================================================
METRIC_COLS = [
    "context_recall",
    "llm_context_precision_with_reference",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]


def _build_metrics():
    return [
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
        Faithfulness(),
        ResponseRelevancy(),
        AnswerCorrectness(),
    ]


def _single_pass(label, dataset, llm, embeddings):
    """단일 Judge LLM으로 평가 수행."""
    print(f"\n🧪 [{label}] 평가 실행 — 샘플 {len(dataset)}개")
    result = evaluate(
        dataset=dataset,
        metrics=_build_metrics(),
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        show_progress=True,
    )
    return result.to_pandas()


def run_evaluation(name, dataset, judge_llm, embeddings):
    """단일 Judge LLM으로 평가 수행."""
    print(f"\n🎯 [{name}] Ragas 평가 시작 — 총 {len(dataset)} 샘플")

    df = _single_pass(name, dataset, judge_llm, embeddings)

    nan_count = int(df[METRIC_COLS].isna().any(axis=1).sum())
    print(f"📉 [{name}] NaN 포함 행: {nan_count}/{len(df)}")

    out_csv = os.path.join(SCRIPT_DIR, f"{name}_ragas_scores.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"💾 CSV 저장: {out_csv}")

    avgs = {c: float(df[c].mean()) for c in METRIC_COLS if c in df.columns and df[c].notna().any()}
    coverage = {c: int(df[c].notna().sum()) for c in METRIC_COLS if c in df.columns}
    print(f"📊 [{name}] 평균 메트릭: {json.dumps(avgs, ensure_ascii=False, indent=2)}")
    print(f"📈 [{name}] 유효 샘플 수 (비-NaN): {coverage}")
    return df, avgs, coverage


# =============================================================================
# Main
# =============================================================================
def main():
    print("🚀 Ragas 평가 환경 구성")

    # 1) 데이터 로드
    golden = load_jsonl(os.path.join(SCRIPT_DIR, "../golden_dataset_v2.jsonl"))
    basic_results = load_json(os.path.join(SCRIPT_DIR, "../week3/RAGresult.json"))
    advanced_results = load_json(os.path.join(SCRIPT_DIR, "../week4/advanced_result.json"))
    print(f"   golden_dataset_v2: {len(golden)}문항")
    print(f"   Basic RAG result: {len(basic_results)}문항")
    print(f"   Advanced RAG result: {len(advanced_results)}문항")

    # 샘플 제한 (RAGAS_LIMIT 환경변수)
    limit_env = os.getenv("RAGAS_LIMIT", "")
    limit = int(limit_env) if limit_env.isdigit() else None
    if limit is not None and limit > 0:
        basic_results = basic_results[:limit]
        advanced_results = advanced_results[:limit]
        print(f"🔬 RAGAS_LIMIT={limit} → 첫 {limit}문항만 평가")

    # 2) Judge LLM + 임베딩
    judge = build_judge_llm()
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    print(f"🧑‍⚖️ Judge LLM: Gemini 2.5 Flash")

    # 3) EvaluationDataset 구성
    basic_ds = build_eval_dataset(golden, basic_results)
    advanced_ds = build_eval_dataset(golden, advanced_results)

    # 4) 평가 실행
    basic_df, basic_avgs, basic_cov = run_evaluation("basic", basic_ds, judge, emb)
    advanced_df, advanced_avgs, advanced_cov = run_evaluation("advanced", advanced_ds, judge, emb)

    # 5) 요약 저장
    summary = {
        "judge_llm": "gemini-2.5-flash",
        "embedding_model": "models/gemini-embedding-001",
        "num_samples": len(basic_results),
        "limit_applied": limit,
        "basic": basic_avgs,
        "basic_coverage": basic_cov,
        "advanced": advanced_avgs,
        "advanced_coverage": advanced_cov,
        "delta": {k: advanced_avgs[k] - basic_avgs.get(k, 0) for k in advanced_avgs if k in basic_avgs},
    }
    out_json = os.path.join(SCRIPT_DIR, "ragas_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 요약 저장: {out_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
