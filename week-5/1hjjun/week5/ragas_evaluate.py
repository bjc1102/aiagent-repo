"""5주차 Ragas 자동 평가 스크립트 (v2 — 교수님 피드백 반영판).

입력:
  - ../golden_dataset_v2.jsonl (ground_truth, ground_truth_contexts 포함)
  - ../week3/RAGresult.json  (Basic RAG 실행 결과)
  - ../week4/advanced_result.json (Advanced RAG 실행 결과)

판정자(Judge LLM): OpenAI gpt-4.1-mini (생성용 Gemini와 다른 model family)
임베딩: Google gemini-embedding-001 (4주차 재사용)

샘플 제한: 환경변수 RAGAS_LIMIT 로 첫 N문항만 평가 (비용 절감).
  예) RAGAS_LIMIT=5 python3 ragas_evaluate.py

메트릭 (총 7개 — LLM 5개 + NonLLM 2개):
  [LLM 기반]
    1) LLMContextRecall  — Retrieval 재현율 (LLM 판정)
    2) LLMContextPrecisionWithReference — Retrieval 정밀도 (LLM 판정)
    3) Faithfulness — Generation 환각 체크
    4) ResponseRelevancy — Generation 질문 관련성
    5) AnswerCorrectness — End-to-end 정답 일치
  [NonLLM 기반 — 교수님 피드백 (b)]
    6) NonLLMContextRecall — retrieved vs reference 임베딩 유사도 기반 Recall
    7) NonLLMContextPrecisionWithReference — 위와 동일한 Precision

출력:
  - basic_ragas_scores.csv / advanced_ragas_scores.csv (문항별 7메트릭)
  - ragas_summary.json (평균 요약)
  - faithfulness_claims.json (Faithfulness 떨어진 claim 리스트 — 교수님 피드백 (c))
"""

import os
import json
import sys
import asyncio
import pandas as pd

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    LLMContextRecall,
    LLMContextPrecisionWithReference,
    Faithfulness,
    ResponseRelevancy,
    AnswerCorrectness,
    NonLLMContextRecall,
    NonLLMContextPrecisionWithReference,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


# .env 로드 (find_dotenv 이슈 회피용 직접 파싱)
def _load_env():
    env_path = os.path.join(PROJECT_DIR, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)


_load_env()


# =============================================================================
# Judge LLM: OpenAI gpt-4.1-mini
# =============================================================================
def build_judge_llm():
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key or openai_key == "your_api_key_here":
        sys.exit("❌ OPENAI_API_KEY 필요 (.env 확인)")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0, timeout=120, max_retries=2)


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
    samples, ids = [], []
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
        ids.append(q_id)
    return EvaluationDataset(samples=samples), ids


# =============================================================================
# 평가 메트릭
# =============================================================================
METRIC_COLS = [
    "context_recall",
    "llm_context_precision_with_reference",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "non_llm_context_recall",
    "non_llm_context_precision_with_reference",
]


def _build_metrics():
    return [
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
        Faithfulness(),
        ResponseRelevancy(),
        AnswerCorrectness(),
        NonLLMContextRecall(),
        NonLLMContextPrecisionWithReference(),
    ]


def run_evaluation(name, dataset, ids, judge_llm, embeddings):
    """평가 실행 + CSV 저장 + 평균 통계."""
    print(f"\n🎯 [{name}] Ragas 평가 시작 — 샘플 {len(dataset)}개")
    result = evaluate(
        dataset=dataset,
        metrics=_build_metrics(),
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        show_progress=True,
    )
    df = result.to_pandas()
    df.insert(0, "id", ids)

    nan_count = int(df[METRIC_COLS].isna().any(axis=1).sum())
    print(f"📉 [{name}] NaN 포함 행: {nan_count}/{len(df)}")

    out_csv = os.path.join(SCRIPT_DIR, f"{name}_ragas_scores.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"💾 CSV 저장: {out_csv}")

    avgs = {c: float(df[c].mean()) for c in METRIC_COLS if c in df.columns and df[c].notna().any()}
    coverage = {c: int(df[c].notna().sum()) for c in METRIC_COLS if c in df.columns}
    print(f"📊 [{name}] 평균: {json.dumps(avgs, ensure_ascii=False, indent=2)}")
    return df, avgs, coverage


# =============================================================================
# Faithfulness claim 진단 (교수님 피드백 (c))
# =============================================================================
async def diagnose_faithfulness(name, dataset, ids, judge_llm):
    """Faithfulness가 추출한 claim과 verdict를 직접 덤프 — 어떤 claim이 떨어졌는지 확인."""
    from ragas.metrics._faithfulness import StatementGeneratorInput, NLIStatementInput

    print(f"\n🔍 [{name}] Faithfulness claim 진단 시작")
    wrapped_llm = LangchainLLMWrapper(judge_llm)
    metric = Faithfulness(llm=wrapped_llm)

    diagnostics = []
    for i, sample in enumerate(dataset):
        try:
            ctx_str = "\n".join(sample.retrieved_contexts) if sample.retrieved_contexts else ""

            # 1) statement 추출
            stmts_input = StatementGeneratorInput(
                question=sample.user_input,
                answer=sample.response,
            )
            stmts_response = await metric.statement_generator_prompt.generate(
                llm=wrapped_llm, data=stmts_input,
            )
            statements = list(stmts_response.statements)

            # 2) NLI 판정
            nli_input = NLIStatementInput(context=ctx_str, statements=statements)
            nli_response = await metric.nli_statements_prompt.generate(
                llm=wrapped_llm, data=nli_input,
            )
            verdicts = []
            for s in nli_response.statements:
                verdicts.append({
                    "statement": s.statement,
                    "reason": s.reason,
                    "verdict": int(s.verdict),
                })
            n_supported = sum(1 for v in verdicts if v["verdict"] == 1)
            n_total = len(verdicts) or 1
            score = n_supported / n_total

            diagnostics.append({
                "id": ids[i],
                "score": round(score, 3),
                "n_total_claims": n_total,
                "n_supported": n_supported,
                "n_unsupported": n_total - n_supported,
                "claims": verdicts,
            })
            print(f"  {ids[i]}: {n_supported}/{n_total} 지지됨 (score {score:.2f})")
        except Exception as e:
            print(f"  ⚠️ {ids[i]} 진단 실패: {type(e).__name__}: {str(e)[:120]}")
            diagnostics.append({"id": ids[i], "error": str(e)[:200]})

    return diagnostics


# =============================================================================
# Main
# =============================================================================
def main():
    print("🚀 Ragas 평가 환경 구성")

    # 1) 데이터 로드
    golden = load_jsonl(os.path.join(PROJECT_DIR, "golden_dataset_v2.jsonl"))
    basic_results = load_json(os.path.join(PROJECT_DIR, "week3", "RAGresult.json"))
    advanced_results = load_json(os.path.join(PROJECT_DIR, "week4", "advanced_result.json"))
    print(f"   golden_dataset_v2: {len(golden)}문항")
    print(f"   Basic RAG result: {len(basic_results)}문항")
    print(f"   Advanced RAG result: {len(advanced_results)}문항")

    limit_env = os.getenv("RAGAS_LIMIT", "")
    limit = int(limit_env) if limit_env.isdigit() else None
    if limit is not None and limit > 0:
        basic_results = basic_results[:limit]
        advanced_results = advanced_results[:limit]
        print(f"🔬 RAGAS_LIMIT={limit} → 첫 {limit}문항만 평가")

    # 2) Judge LLM + 임베딩
    judge = build_judge_llm()
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    print(f"🧑‍⚖️ Judge LLM: OpenAI gpt-4.1-mini")

    # 3) Dataset 구성
    basic_ds, basic_ids = build_eval_dataset(golden, basic_results)
    advanced_ds, advanced_ids = build_eval_dataset(golden, advanced_results)

    # 4) 평가 실행 (LLM + NonLLM 메트릭)
    basic_df, basic_avgs, basic_cov = run_evaluation("basic", basic_ds, basic_ids, judge, emb)
    advanced_df, advanced_avgs, advanced_cov = run_evaluation("advanced", advanced_ds, advanced_ids, judge, emb)

    # 5) Faithfulness claim 진단
    basic_diag = asyncio.run(diagnose_faithfulness("basic", basic_ds, basic_ids, judge))
    advanced_diag = asyncio.run(diagnose_faithfulness("advanced", advanced_ds, advanced_ids, judge))
    diag_path = os.path.join(SCRIPT_DIR, "faithfulness_claims.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(
            {"basic": basic_diag, "advanced": advanced_diag},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n🔬 Faithfulness 진단 저장: {diag_path}")

    # 6) 요약 저장
    summary = {
        "judge_llm": "gpt-4.1-mini",
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
