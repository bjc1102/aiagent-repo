"""
Step 2 — Ragas evaluation pipeline (v0.2+ schema).

INPUT:  rag_outputs_basic.json, rag_outputs_advanced.json (from Step 1)
OUTPUT: basic_ragas_scores.csv, advanced_ragas_scores.csv
        + comparison_summary.json

METRICS (5):
  1. ContextRecall                       — Retrieval — uses `reference_contexts`/`reference`
  2. LLMContextPrecisionWithReference    — Retrieval — uses `reference`
  3. Faithfulness                        — Generation — uses `response`+`retrieved_contexts`
  4. ResponseRelevancy                   — Generation — uses `response`+`user_input`
  5. AnswerCorrectness                   — End-to-end  — uses `reference`+`response`

EVALUATOR LLM:
  Claude Sonnet 4.5 (claude-sonnet-4-5) — different *version* (not family) from the
  generation model (claude-sonnet-4 in week-4). The TASK recommends a different
  family, but we only have ANTHROPIC_API_KEY available locally. Documenting this
  trade-off in README.md.

EMBEDDINGS:
  multilingual-e5-small (HuggingFace) — same as week-4 vector store.
  Reused for AnswerCorrectness/ResponseRelevancy semantic similarity.

KOREAN PROMPTS:
  Ragas internal prompts are English by default → adapt_prompts(language="korean")
  on every metric instance, then set_prompts(**adapted) so the LLM judge sees
  Korean prompts (matches the Korean PDF domain).

COST:
  ~15 questions × 5 metrics × 2 pipelines = 150 metric runs.
  Each metric calls the LLM 1–3 times → ~300–500 LLM calls total.
  At Claude Sonnet 4.5 prices that's roughly $3–8.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    LLMContextRecall,
    LLMContextPrecisionWithReference,
    Faithfulness,
    ResponseRelevancy,
    AnswerCorrectness,
)

# ============================================================
# Config
# ============================================================
HERE = Path(__file__).resolve().parent
BASIC_INPUT = HERE / "rag_outputs_basic.json"
ADVANCED_INPUT = HERE / "rag_outputs_advanced.json"

EVALUATOR_MODEL = "claude-sonnet-4-5"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Load API key from 1week/.env if not already in env
if not os.environ.get("ANTHROPIC_API_KEY"):
    env_path = HERE.parent.parent / "1week" / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.strip().split("=", 1)[1]


# ============================================================
# Build SingleTurnSample list from rag_outputs_*.json
# ============================================================
def build_dataset(rag_outputs_path: Path) -> EvaluationDataset:
    with open(rag_outputs_path, encoding="utf-8") as f:
        rows = json.load(f)

    samples = []
    for r in rows:
        # v0.2+ schema:
        #   question      → user_input
        #   answer        → response
        #   contexts      → retrieved_contexts
        #   ground_truth  → reference
        #   ground_truth_contexts → reference_contexts
        samples.append(
            SingleTurnSample(
                user_input=r["question"],
                response=r["response"],
                retrieved_contexts=r["retrieved_contexts"],
                reference=r["ground_truth"],
                reference_contexts=r["ground_truth_contexts"],
            )
        )
    return EvaluationDataset(samples=samples), rows


# ============================================================
# Build evaluator LLM/embeddings + Korean-localised metrics
# ============================================================
def build_evaluator():
    """
    Wrap LangChain LLM/embeddings with Ragas wrappers.
    Build metric *instances* (class-form is the v0.2+ standard;
    snake_case function-form is deprecation-marked).
    """
    eval_llm = LangchainLLMWrapper(
        ChatAnthropic(model=EVALUATOR_MODEL, temperature=0, max_tokens=2000)
    )
    eval_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    metrics = [
        LLMContextRecall(llm=eval_llm),
        LLMContextPrecisionWithReference(llm=eval_llm),
        Faithfulness(llm=eval_llm),
        ResponseRelevancy(llm=eval_llm, embeddings=eval_emb),
        AnswerCorrectness(llm=eval_llm, embeddings=eval_emb),
    ]

    return eval_llm, eval_emb, metrics


def adapt_metrics_to_korean(metrics, eval_llm):
    """
    Ragas internal prompts are English. We translate them to Korean once
    so the evaluator LLM judges Korean responses against Korean prompts —
    fewer translation artefacts in the score.

    NOTE (Ragas 0.2.15): `adapt_prompts` is async — it returns a coroutine.
    Synchronous `m.set_prompts(**coroutine)` raises:
        "argument after ** must be a mapping, not coroutine"
    So we await via asyncio.run().
    """
    for m in metrics:
        try:
            coro = m.adapt_prompts(language="korean", llm=eval_llm)
            adapted = asyncio.run(coro) if asyncio.iscoroutine(coro) else coro
            m.set_prompts(**adapted)
            print(f"  [korean prompts] {m.__class__.__name__} ✓")
        except Exception as e:
            print(f"  [korean prompts] {m.__class__.__name__} ✗ ({e})")


# ============================================================
# Evaluate one pipeline
# ============================================================
def evaluate_pipeline(name: str, rag_outputs_path: Path,
                     metrics, eval_llm, eval_emb, out_csv: Path):
    print(f"\n{'='*70}\n Evaluating: {name}\n{'='*70}")
    if not rag_outputs_path.exists():
        print(f"  ! {rag_outputs_path} missing — run 01_run_rag_and_collect.py first.")
        return None

    dataset, rows = build_dataset(rag_outputs_path)

    print(f"  Dataset size: {len(dataset)} samples")
    print(f"  Running 5 metrics through {EVALUATOR_MODEL} …")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_emb,
        show_progress=True,
    )

    df = result.to_pandas()
    # Tag rows with id / difficulty / source_year so we can group later.
    df["id"] = [r["id"] for r in rows]
    df["difficulty"] = [r["difficulty"] for r in rows]
    df["source_year"] = [r["source_year"] for r in rows]

    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"  saved → {out_csv}")

    # Print mean per metric
    metric_cols = [c for c in df.columns
                   if c not in {"user_input","response","retrieved_contexts",
                                "reference","reference_contexts",
                                "id","difficulty","source_year"}]
    print("\n  Mean metric scores:")
    for c in metric_cols:
        try:
            print(f"    {c:35s}  {df[c].astype(float).mean():.4f}")
        except Exception:
            pass

    return df, metric_cols


# ============================================================
# Main
# ============================================================
def main():
    eval_llm, eval_emb, metrics = build_evaluator()
    print("Adapting metric prompts to Korean…")
    adapt_metrics_to_korean(metrics, eval_llm)

    basic = evaluate_pipeline(
        "Basic RAG", BASIC_INPUT, metrics, eval_llm, eval_emb,
        HERE / "basic_ragas_scores.csv",
    )
    advanced = evaluate_pipeline(
        "Advanced RAG", ADVANCED_INPUT, metrics, eval_llm, eval_emb,
        HERE / "advanced_ragas_scores.csv",
    )

    if basic and advanced:
        b_df, m_cols = basic
        a_df, _ = advanced
        summary = {
            "evaluator_model": EVALUATOR_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "n_samples": int(len(b_df)),
            "metrics": {},
        }
        for c in m_cols:
            try:
                summary["metrics"][c] = {
                    "basic_mean": float(b_df[c].astype(float).mean()),
                    "advanced_mean": float(a_df[c].astype(float).mean()),
                    "delta": float(a_df[c].astype(float).mean() - b_df[c].astype(float).mean()),
                }
            except Exception:
                pass
        with open(HERE / "comparison_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nsaved → {HERE/'comparison_summary.json'}")


if __name__ == "__main__":
    # Ragas uses asyncio internally; on macOS the default is fine.
    main()
