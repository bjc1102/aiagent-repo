from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    AnswerCorrectness,
    ContextRecall,
    Faithfulness,
    LLMContextPrecisionWithReference,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig


ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def load_pipeline_dataset(
    dataset_rows: dict[str, dict],
    score_csv: Path,
    top_contexts: int,
    max_context_chars: int,
) -> tuple[EvaluationDataset, list[str]]:
    score_rows = pd.read_csv(score_csv)
    samples = []
    ids = []
    for _, row in score_rows.iterrows():
        golden = dataset_rows[row["id"]]
        ids.append(row["id"])
        retrieved = [item["text"][:max_context_chars] for item in json.loads(row["retrieved_contexts"])]
        samples.append(
            SingleTurnSample(
                user_input=golden["question"],
                response=row["response"],
                retrieved_contexts=retrieved[:top_contexts],
                reference=golden["ground_truth"],
                reference_contexts=golden["ground_truth_contexts"],
            )
        )
    return EvaluationDataset(samples=samples), ids


def make_google_clients(chat_model: str, embedding_model: str, max_tokens: int) -> tuple[ChatOpenAI, OpenAIEmbeddings]:
    api_key = os.getenv("GOOGLE_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("GOOGLE_API_KEY and BASE_URL must be set in .env")

    llm = ChatOpenAI(
        model=chat_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=max_tokens,
        timeout=120,
    )
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,
        timeout=120,
    )
    return llm, embeddings


def run_pipeline(
    name: str,
    dataset: EvaluationDataset,
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    output_dir: Path,
    batch_size: int,
    ids: list[str],
) -> pd.DataFrame:
    metrics = [
        ContextRecall(),
        LLMContextPrecisionWithReference(),
        Faithfulness(),
        ResponseRelevancy(),
        AnswerCorrectness(),
    ]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=180, max_retries=2, max_wait=30),
        raise_exceptions=False,
        show_progress=False,
        batch_size=batch_size,
    )
    frame = result.to_pandas()
    frame.insert(0, "id", ids)
    frame.to_csv(output_dir / f"{name}_ragas_google_scores.csv", index=False, encoding="utf-8-sig")
    return frame


def summarize(
    basic: pd.DataFrame,
    advanced: pd.DataFrame,
    output_dir: Path,
    chat_model: str,
    embedding_model: str,
) -> None:
    metric_aliases = {
        "context_recall": "Context Recall",
        "llm_context_precision_with_reference": "Context Precision",
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "answer_correctness": "Answer Correctness",
    }
    rows = []
    for column, label in metric_aliases.items():
        if column not in basic.columns or column not in advanced.columns:
            continue
        basic_mean = float(pd.to_numeric(basic[column], errors="coerce").mean())
        advanced_mean = float(pd.to_numeric(advanced[column], errors="coerce").mean())
        rows.append(
            {
                "metric": label,
                "basic": round(basic_mean, 4),
                "advanced": round(advanced_mean, 4),
                "delta": round(advanced_mean - basic_mean, 4),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "ragas_google_summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "model": chat_model,
        "embedding_model": embedding_model,
        "samples": {"basic": len(basic), "advanced": len(advanced)},
        "metric_means": rows,
    }
    (output_dir / "ragas_google_evaluation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run real Ragas evaluation with Google OpenAI-compatible API.")
    parser.add_argument("--dataset", type=Path, default=SUBMISSION_DIR / "golden_dataset_v2.jsonl")
    parser.add_argument("--basic", type=Path, default=SUBMISSION_DIR / "basic_ragas_scores.csv")
    parser.add_argument("--advanced", type=Path, default=SUBMISSION_DIR / "advanced_ragas_scores.csv")
    parser.add_argument("--output-dir", type=Path, default=SUBMISSION_DIR)
    parser.add_argument("--pipeline", choices=["basic", "advanced", "both"], default="both")
    parser.add_argument("--chat-model", default=os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash-lite"))
    parser.add_argument("--embedding-model", default=os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-contexts", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=700)
    args = parser.parse_args()

    dataset_rows = load_jsonl(args.dataset)
    llm, embeddings = make_google_clients(args.chat_model, args.embedding_model, args.max_tokens)

    if args.pipeline in {"basic", "both"}:
        basic_dataset, basic_ids = load_pipeline_dataset(dataset_rows, args.basic, args.top_contexts, args.max_context_chars)
        basic_frame = run_pipeline("basic", basic_dataset, llm, embeddings, args.output_dir, args.batch_size, basic_ids)
    elif (args.output_dir / "basic_ragas_google_scores.csv").exists():
        basic_frame = pd.read_csv(args.output_dir / "basic_ragas_google_scores.csv")
    else:
        basic_frame = pd.DataFrame()

    if args.pipeline in {"advanced", "both"}:
        advanced_dataset, advanced_ids = load_pipeline_dataset(dataset_rows, args.advanced, args.top_contexts, args.max_context_chars)
        advanced_frame = run_pipeline("advanced", advanced_dataset, llm, embeddings, args.output_dir, args.batch_size, advanced_ids)
    elif (args.output_dir / "advanced_ragas_google_scores.csv").exists():
        advanced_frame = pd.read_csv(args.output_dir / "advanced_ragas_google_scores.csv")
    else:
        advanced_frame = pd.DataFrame()

    if not basic_frame.empty and not advanced_frame.empty:
        summarize(basic_frame, advanced_frame, args.output_dir, args.chat_model, args.embedding_model)


if __name__ == "__main__":
    main()
