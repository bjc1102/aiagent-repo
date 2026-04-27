import argparse
import asyncio
import csv
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import BasicRag
import HybridCom
import HybridComRerank
import HybridRerank


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = BASE_DIR / "golden_dataset_step2_pilot_5.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "result" / "step2_pilot_5"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
BASE_SAMPLE_COLUMNS = {
    "user_input",
    "retrieved_contexts",
    "reference_contexts",
    "response",
    "reference",
    "reference_context_ids",
    "rubric",
    "multi_responses",
    "persona_name",
    "query_style",
    "query_length",
}
DISPLAY_NAME_MAP = {
    "context_recall": "Context Recall",
    "llm_context_precision_with_reference": "Context Precision",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "answer_correctness": "Answer Correctness",
    "year_accuracy": "Year Accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ragas evaluation for week5 Basic/Advanced RAG pipelines."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to golden dataset JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store csv/jsonl outputs.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=["basic", "advanced", "hybrid_com", "hybrid_com_rerank"],
        default=["basic", "advanced"],
        help="Pipelines to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions for pilot runs.",
    )
    parser.add_argument(
        "--basic-k",
        type=int,
        default=4,
        help="Top-k for Basic RAG retrieval.",
    )
    parser.add_argument(
        "--advanced-top-n",
        type=int,
        default=HybridRerank.RERANK_TOP,
        help="Final top-n chunks for Advanced RAG.",
    )
    parser.add_argument(
        "--hybrid-com-top-n",
        type=int,
        default=HybridCom.COMPRESSION_TOP,
        help="Final top-n chunks for Hybrid + Contextual Compression.",
    )
    parser.add_argument(
        "--hybrid-com-rerank-top-n",
        type=int,
        default=HybridComRerank.COMPRESSION_RERANK_TOP,
        help="Final top-n chunks for Hybrid + Contextual Compression + Rerank.",
    )
    parser.add_argument(
        "--evaluator-provider",
        choices=["auto", "openai", "anthropic"],
        default="openai",
        help="Evaluator LLM provider. openai is the safe default; auto prefers Anthropic when configured.",
    )
    parser.add_argument(
        "--korean-prompts",
        action="store_true",
        help="Adapt Ragas metric prompts to Korean before evaluation.",
    )
    parser.add_argument(
        "--rebuild-vectorstore",
        action="store_true",
        help="Rebuild Chroma DB before running evaluation.",
    )
    parser.add_argument(
        "--with-year-accuracy",
        action="store_true",
        help="Add custom Year Accuracy metric using source_year metadata.",
    )
    return parser.parse_args()


def load_jsonl(path: Path, limit: Optional[int] = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    return rows


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_vectorstore(rebuild: bool):
    has_existing_db = BasicRag.CHROMA_DIR.exists() and any(BasicRag.CHROMA_DIR.iterdir())
    if rebuild or not has_existing_db:
        print("Building Chroma vectorstore for evaluation...")
        return BasicRag.build_vectorstore()

    print("Loading existing Chroma vectorstore...")
    return BasicRag.load_vectorstore()


def import_ragas_components():
    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except (ImportError, TypeError) as exc:
        raise SystemExit(
            "Ragas evaluation dependencies are missing. "
            "Install them first with `python3 -m pip install -r week5/monkana/requirements-ragas.txt`."
        ) from exc

    return {
        "evaluate": evaluate,
        "EvaluationDataset": EvaluationDataset,
        "SingleTurnSample": SingleTurnSample,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "AnswerCorrectness": AnswerCorrectness,
        "Faithfulness": Faithfulness,
        "LLMContextPrecisionWithReference": LLMContextPrecisionWithReference,
        "LLMContextRecall": LLMContextRecall,
        "ResponseRelevancy": ResponseRelevancy,
    }


def create_year_accuracy_metric():
    from dataclasses import dataclass, field

    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics.base import MetricOutputType, MetricType, SingleTurnMetric

    stopwords = {
        "입니다",
        "있습니다",
        "경우",
        "기준",
        "적용",
        "적용됩니다",
        "변경",
        "변경되었습니다",
        "본인부담",
        "본인부담금",
        "본인부담률",
        "의료급여",
        "수급권자",
        "환자",
        "그리고",
        "각각",
        "해당",
        "포함됩니다",
        "대상",
    }

    def normalize_text(text: str) -> str:
        return " ".join((text or "").replace("\n", " ").split())

    def split_years(source_year: str) -> list[str]:
        return [year.strip() for year in (source_year or "").split(",") if year.strip()]

    def find_year_positions(text: str, years: list[str]) -> list[tuple[str, int]]:
        positions = []
        for year in years:
            idx = text.find(f"{year}년")
            if idx >= 0:
                positions.append((year, idx))
        return sorted(positions, key=lambda item: item[1])

    def extract_year_segments(text: str, years: list[str]) -> dict[str, str]:
        clean = normalize_text(text)
        positions = find_year_positions(clean, years)
        segments: dict[str, str] = {}
        if not positions:
            return segments

        for index, (year, start) in enumerate(positions):
            end = positions[index + 1][1] if index + 1 < len(positions) else len(clean)
            segments[year] = clean[start:end].strip(" ,.")
        return segments

    def extract_tokens(text: str) -> set[str]:
        import re

        clean = normalize_text(text)
        patterns = []
        patterns.extend(re.findall(r"KTAS\s*\d", clean, flags=re.IGNORECASE))
        patterns.extend(re.findall(r"\d[\d,]*(?:\.\d+)?%", clean))
        patterns.extend(re.findall(r"\d[\d,]*(?:\.\d+)?원", clean))
        patterns.extend(re.findall(r"[가-힣]{2,}", clean))

        normalized = set()
        for token in patterns:
            token_norm = token.replace(" ", "").strip(" ,.")
            if token_norm and token_norm not in stopwords:
                normalized.add(token_norm)
        return normalized

    def contains_expected_token(segment_text: str, token: str) -> bool:
        compact = normalize_text(segment_text).replace(" ", "")
        if any(marker in token for marker in ["%", "원", "KTAS"]):
            return token.replace(" ", "") in compact
        return token in compact

    def token_match_ratio(segment_text: str, expected_tokens: set[str]) -> float:
        if not expected_tokens:
            return 1.0
        matched = sum(
            1 for token in expected_tokens if contains_expected_token(segment_text, token)
        )
        return matched / len(expected_tokens)

    def build_expected_tokens(
        years: list[str],
        reference: str,
        reference_contexts: list[str],
    ) -> dict[str, set[str]]:
        reference_segments = extract_year_segments(reference, years)
        expected: dict[str, set[str]] = {}

        for index, year in enumerate(years):
            tokens = extract_tokens(reference_segments.get(year, ""))
            if not tokens and index < len(reference_contexts):
                tokens = extract_tokens(reference_contexts[index])
            expected[year] = tokens

        if len(years) > 1:
            common_tokens = set.intersection(*(tokens for tokens in expected.values() if tokens))
            if common_tokens:
                for year in years:
                    reduced = expected[year] - common_tokens
                    if reduced:
                        expected[year] = reduced

        return expected

    @dataclass
    class YearAccuracy(SingleTurnMetric):
        name: str = "year_accuracy"
        _required_columns: dict = field(
            default_factory=lambda: {
                MetricType.SINGLE_TURN: {
                    "user_input",
                    "response",
                    "query_style",
                    "reference",
                    "reference_contexts",
                }
            }
        )
        output_type: Optional[MetricOutputType] = MetricOutputType.BINARY

        def init(self, run_config) -> None:
            return None

        async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks) -> float:
            row = sample.to_dict()
            response = normalize_text(row.get("response") or "")
            reference = normalize_text(row.get("reference") or "")
            source_year = row.get("query_style") or ""
            reference_contexts = row.get("reference_contexts") or []
            years = split_years(source_year)

            if not years:
                return 0.0

            if len(years) == 1:
                only_year = years[0]
                other_years = {"2024", "2025", "2026", "2027"} - {only_year}
                if any(f"{year}년" in response for year in other_years):
                    return 0.0
                expected_tokens = build_expected_tokens(years, reference, reference_contexts)[only_year]
                if not expected_tokens:
                    return 1.0
                return float(token_match_ratio(response, expected_tokens) >= 0.5)

            response_order = [year for year, _ in find_year_positions(response, years)]
            if response_order != years:
                return 0.0

            response_segments = extract_year_segments(response, years)
            if any(year not in response_segments for year in years):
                return 0.0

            expected_tokens = build_expected_tokens(years, reference, reference_contexts)
            for year in years:
                needed = expected_tokens.get(year, set())
                segment_text = response_segments.get(year, "")
                if needed and token_match_ratio(segment_text, needed) < 0.5:
                    return 0.0

            return 1.0

    return YearAccuracy


def choose_evaluator_components(
    provider: str,
    llm_wrapper_cls,
    embedding_wrapper_cls,
):
    if provider in {"auto", "anthropic"} and os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            if provider == "anthropic":
                raise SystemExit(
                    "ANTHROPIC_API_KEY is set but `langchain-anthropic` is not installed."
                )
        else:
            model_name = os.getenv("ANTHROPIC_EVAL_MODEL", "claude-sonnet-4-5")
            evaluator_llm = llm_wrapper_cls(
                ChatAnthropic(model=model_name, temperature=0)
            )
            evaluator_embeddings = embedding_wrapper_cls(
                OpenAIEmbeddings(
                    model=os.getenv(
                        "OPENAI_EVAL_EMBEDDING_MODEL",
                        DEFAULT_EMBEDDING_MODEL,
                    )
                )
            )
            return evaluator_llm, evaluator_embeddings, {
                "provider": "anthropic",
                "llm_model": model_name,
                "embedding_model": os.getenv(
                    "OPENAI_EVAL_EMBEDDING_MODEL",
                    DEFAULT_EMBEDDING_MODEL,
                ),
            }

    if provider == "anthropic":
        raise SystemExit(
            "Anthropic evaluator was requested, but `ANTHROPIC_API_KEY` is not configured in .env."
        )

    model_name = os.getenv("OPENAI_EVAL_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    embedding_model = os.getenv("OPENAI_EVAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    evaluator_llm = llm_wrapper_cls(ChatOpenAI(model=model_name, temperature=0))
    evaluator_embeddings = embedding_wrapper_cls(OpenAIEmbeddings(model=embedding_model))
    return evaluator_llm, evaluator_embeddings, {
        "provider": "openai",
        "llm_model": model_name,
        "embedding_model": embedding_model,
    }


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def adapt_metric_prompts(metrics: list[Any], evaluator_llm: Any, language: str) -> list[str]:
    logs: list[str] = []
    for metric in metrics:
        metric_name = type(metric).__name__
        if hasattr(metric, "prompt") and hasattr(metric.prompt, "adapt"):
            adapted_prompt = run_async(
                metric.prompt.adapt(
                    target_language=language,
                    llm=evaluator_llm,
                    adapt_instruction=True,
                )
            )
            metric.prompt = adapted_prompt
            logs.append(f"{metric_name}: prompt.adapt -> {getattr(metric.prompt, 'language', language)}")
            continue

        if hasattr(metric, "adapt_prompts") and hasattr(metric, "set_prompts"):
            adapted = run_async(metric.adapt_prompts(language=language, llm=evaluator_llm))
            metric.set_prompts(**adapted)
            logs.append(f"{metric_name}: adapt_prompts/set_prompts")
            continue

        logs.append(f"{metric_name}: prompt adaptation hook not found")
    return logs


def make_metrics(components: dict[str, Any], include_year_accuracy: bool = False) -> list[Any]:
    metrics = [
        components["LLMContextRecall"](),
        components["LLMContextPrecisionWithReference"](),
        components["Faithfulness"](),
        components["ResponseRelevancy"](),
        components["AnswerCorrectness"](),
    ]
    if include_year_accuracy:
        metrics.append(create_year_accuracy_metric()())
    return metrics


def serialize_doc(doc: Any) -> dict[str, Any]:
    return {
        "page_content": doc.page_content,
        "metadata": dict(doc.metadata),
    }


def build_basic_runner(vectorstore: Any, top_k: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _run(row: dict[str, Any]) -> dict[str, Any]:
        return BasicRag.run_pipeline(
            vectorstore,
            row["question"],
            k=top_k,
            source_year=row.get("source_year", ""),
        )

    return _run


def build_advanced_runner(vectorstore: Any, all_documents: list[Any], top_n: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _run(row: dict[str, Any]) -> dict[str, Any]:
        return HybridRerank.run_pipeline(
            vectorstore,
            all_documents,
            row["question"],
            source_year=row.get("source_year", ""),
            top_n=top_n,
        )

    return _run


def build_hybrid_com_runner(vectorstore: Any, all_documents: list[Any], top_n: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _run(row: dict[str, Any]) -> dict[str, Any]:
        return HybridCom.run_pipeline(
            vectorstore,
            all_documents,
            row["question"],
            source_year=row.get("source_year", ""),
            top_n=top_n,
        )

    return _run


def build_hybrid_com_rerank_runner(vectorstore: Any, all_documents: list[Any], top_n: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _run(row: dict[str, Any]) -> dict[str, Any]:
        return HybridComRerank.run_pipeline(
            vectorstore,
            all_documents,
            row["question"],
            source_year=row.get("source_year", ""),
            top_n=top_n,
        )

    return _run


def build_evaluation_dataset(
    rows: list[dict[str, Any]],
    runner: Callable[[dict[str, Any]], dict[str, Any]],
    sample_cls,
    dataset_cls,
) -> tuple[Any, list[dict[str, Any]]]:
    samples = []
    traces = []

    for row in rows:
        pipeline_result = runner(row)
        sample = sample_cls(
            user_input=row["question"],
            response=pipeline_result["response"],
            retrieved_contexts=pipeline_result["retrieved_contexts"],
            reference=row["ground_truth"],
            reference_contexts=row.get("ground_truth_contexts", []),
            query_style=row.get("source_year", ""),
        )
        samples.append(sample)
        traces.append(
            {
                "id": row.get("id"),
                "difficulty": row.get("difficulty"),
                "source_year": row.get("source_year"),
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "ground_truth_contexts": row.get("ground_truth_contexts", []),
                "response": pipeline_result["response"],
                "retrieved_contexts": pipeline_result["retrieved_contexts"],
                "retrieved_docs": [serialize_doc(doc) for doc in pipeline_result["retrieved_docs"]],
            }
        )

    return dataset_cls(samples=samples), traces


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def attach_metadata(df: Any, rows: list[dict[str, Any]]):
    df.insert(0, "id", [row.get("id") for row in rows])
    df.insert(1, "difficulty", [row.get("difficulty") for row in rows])
    df.insert(2, "source_year", [row.get("source_year") for row in rows])
    return df


def extract_metric_columns(df: Any) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in BASE_SAMPLE_COLUMNS
        and column not in {"id", "difficulty", "source_year"}
        and str(df[column].dtype) != "object"
    ]


def save_summary_csv(
    path: Path,
    per_pipeline_frames: dict[str, Any],
    metric_columns: list[str],
    pipeline_order: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", *pipeline_order])
        for metric in metric_columns:
            row = [metric]
            for pipeline_name in pipeline_order:
                value = None
                if (
                    pipeline_name in per_pipeline_frames
                    and metric in per_pipeline_frames[pipeline_name].columns
                ):
                    value = float(per_pipeline_frames[pipeline_name][metric].mean())
                row.append(value)
            writer.writerow(row)


def metric_label(metric_name: str) -> str:
    return DISPLAY_NAME_MAP.get(metric_name, metric_name)


def shorten_text(text: Any, limit: int = 80) -> str:
    value = str(text).replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def rounded_or_blank(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{numeric:.{digits}f}"


def build_readable_pipeline_rows(df: Any, metric_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        formatted = {
            "id": row["id"],
            "difficulty": row["difficulty"],
            "source_year": row["source_year"],
            "question": row["user_input"],
            "response": row["response"],
            "reference": row["reference"],
        }
        for metric in metric_columns:
            formatted[metric] = row.get(metric)
        rows.append(formatted)
    return rows


def save_readable_pipeline_csv(path: Path, df: Any, metric_columns: list[str]) -> None:
    rows = build_readable_pipeline_rows(df, metric_columns)
    fieldnames = [
        "id",
        "difficulty",
        "source_year",
        "question",
        "response",
        "reference",
        *metric_columns,
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_comparison_csv(
    path: Path,
    per_pipeline_frames: dict[str, Any],
    metric_columns: list[str],
    pipeline_order: list[str],
) -> None:
    if len(pipeline_order) != 2:
        return

    left_name, right_name = pipeline_order
    left_df = per_pipeline_frames.get(left_name)
    right_df = per_pipeline_frames.get(right_name)
    if left_df is None or right_df is None:
        return

    merged = left_df.merge(
        right_df,
        on=["id", "difficulty", "source_year"],
        suffixes=(f"_{left_name}", f"_{right_name}"),
    )

    fieldnames = [
        "id",
        "difficulty",
        "source_year",
        "question",
        "reference",
        f"response_{left_name}",
        f"response_{right_name}",
    ]
    for metric in metric_columns:
        fieldnames.extend(
            [
                f"{metric}_{left_name}",
                f"{metric}_{right_name}",
                f"{metric}_delta",
            ]
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, row in merged.iterrows():
            output_row = {
                "id": row["id"],
                "difficulty": row["difficulty"],
                "source_year": row["source_year"],
                "question": row[f"user_input_{left_name}"],
                "reference": row[f"reference_{left_name}"],
                f"response_{left_name}": row[f"response_{left_name}"],
                f"response_{right_name}": row[f"response_{right_name}"],
            }
            for metric in metric_columns:
                left_value = row.get(f"{metric}_{left_name}")
                right_value = row.get(f"{metric}_{right_name}")
                output_row[f"{metric}_{left_name}"] = left_value
                output_row[f"{metric}_{right_name}"] = right_value
                if left_value is None or right_value is None:
                    output_row[f"{metric}_delta"] = ""
                else:
                    output_row[f"{metric}_delta"] = float(right_value) - float(left_value)
            writer.writerow(output_row)


def save_overview_markdown(
    path: Path,
    per_pipeline_frames: dict[str, Any],
    metric_columns: list[str],
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# Ragas Result Overview")
    lines.append("")
    lines.append("## Run Setup")
    lines.append("")
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- pipelines: `{', '.join(args.pipelines)}`")
    lines.append(f"- limit: `{args.limit if args.limit is not None else 'all'}`")
    lines.append(f"- basic_k: `{args.basic_k}`")
    lines.append(f"- advanced_top_n: `{args.advanced_top_n}`")
    lines.append(f"- hybrid_com_top_n: `{args.hybrid_com_top_n}`")
    lines.append(f"- hybrid_com_rerank_top_n: `{args.hybrid_com_rerank_top_n}`")
    lines.append(f"- korean_prompts: `{args.korean_prompts}`")
    lines.append("")
    lines.append("## Average Scores")
    lines.append("")
    average_headers = ["Metric", *args.pipelines]
    lines.append("| " + " | ".join(average_headers) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * len(args.pipelines)) + "|")
    for metric in metric_columns:
        row_values = [metric_label(metric)]
        for pipeline_name in args.pipelines:
            value = None
            if (
                pipeline_name in per_pipeline_frames
                and metric in per_pipeline_frames[pipeline_name].columns
            ):
                value = float(per_pipeline_frames[pipeline_name][metric].mean())
            row_values.append(rounded_or_blank(value))
        lines.append("| " + " | ".join(row_values) + " |")

    if len(args.pipelines) == 2:
        left_name, right_name = args.pipelines
        left_df = per_pipeline_frames.get(left_name)
        right_df = per_pipeline_frames.get(right_name)
    else:
        left_df = None
        right_df = None

    if left_df is not None and right_df is not None:
        merged = left_df.merge(
            right_df,
            on=["id", "difficulty", "source_year"],
            suffixes=(f"_{left_name}", f"_{right_name}"),
        )
        lines.append("")
        lines.append("## Per-Question Comparison")
        lines.append("")
        lines.append(
            f"| ID | Difficulty | Year | {left_name} Ans | {right_name} Ans | Ctx Recall {left_name}/{right_name} | Ctx Precision {left_name}/{right_name} | Faithfulness {left_name}/{right_name} | Ans Relevancy {left_name}/{right_name} | Ans Correctness {left_name}/{right_name} |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, row in merged.iterrows():
            lines.append(
                "| {id} | {difficulty} | {source_year} | {basic_ans} | {advanced_ans} | {ctx_recall} | {ctx_precision} | {faithfulness} | {relevancy} | {correctness} |".format(
                    id=row["id"],
                    difficulty=row["difficulty"],
                    source_year=row["source_year"],
                    basic_ans=shorten_text(row[f"response_{left_name}"], 24),
                    advanced_ans=shorten_text(row[f"response_{right_name}"], 24),
                    ctx_recall=f"{rounded_or_blank(row.get(f'context_recall_{left_name}'))}/{rounded_or_blank(row.get(f'context_recall_{right_name}'))}",
                    ctx_precision=f"{rounded_or_blank(row.get(f'llm_context_precision_with_reference_{left_name}'))}/{rounded_or_blank(row.get(f'llm_context_precision_with_reference_{right_name}'))}",
                    faithfulness=f"{rounded_or_blank(row.get(f'faithfulness_{left_name}'))}/{rounded_or_blank(row.get(f'faithfulness_{right_name}'))}",
                    relevancy=f"{rounded_or_blank(row.get(f'answer_relevancy_{left_name}'))}/{rounded_or_blank(row.get(f'answer_relevancy_{right_name}'))}",
                    correctness=f"{rounded_or_blank(row.get(f'answer_correctness_{left_name}'))}/{rounded_or_blank(row.get(f'answer_correctness_{right_name}'))}",
                )
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_run_metadata(path: Path, args: argparse.Namespace, evaluator_meta: dict[str, Any], prompt_logs: list[str]) -> None:
    metadata = {
        "dataset": str(args.dataset),
        "pipelines": args.pipelines,
        "limit": args.limit,
        "basic_k": args.basic_k,
        "advanced_top_n": args.advanced_top_n,
        "hybrid_com_top_n": args.hybrid_com_top_n,
        "hybrid_com_rerank_top_n": args.hybrid_com_rerank_top_n,
        "rebuild_vectorstore": args.rebuild_vectorstore,
        "korean_prompts": args.korean_prompts,
        "with_year_accuracy": args.with_year_accuracy,
        "evaluator": evaluator_meta,
        "ragas_version": safe_version("ragas"),
        "langchain_openai_version": safe_version("langchain-openai"),
        "prompt_adaptation": prompt_logs,
    }
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    output_dir = ensure_output_dir(args.output_dir)
    rows = load_jsonl(args.dataset, limit=args.limit)
    if not rows:
        raise SystemExit("No evaluation rows found in dataset.")

    components = import_ragas_components()
    evaluator_llm, evaluator_embeddings, evaluator_meta = choose_evaluator_components(
        provider=args.evaluator_provider,
        llm_wrapper_cls=components["LangchainLLMWrapper"],
        embedding_wrapper_cls=components["LangchainEmbeddingsWrapper"],
    )

    vectorstore = ensure_vectorstore(rebuild=args.rebuild_vectorstore)
    all_documents = None
    pipelines_requiring_bm25 = {"advanced", "hybrid_com", "hybrid_com_rerank"}
    if any(name in pipelines_requiring_bm25 for name in args.pipelines):
        print("Loading BM25 documents for hybrid-style pipelines...")
        all_documents = HybridRerank.load_all_documents()

    prompt_logs: list[str] = []
    per_pipeline_frames: dict[str, Any] = {}
    all_metric_columns: set[str] = set()

    pipeline_runners = {
        "basic": build_basic_runner(vectorstore, top_k=args.basic_k),
    }
    if all_documents is not None:
        pipeline_runners["advanced"] = build_advanced_runner(
            vectorstore,
            all_documents,
            top_n=args.advanced_top_n,
        )
        pipeline_runners["hybrid_com"] = build_hybrid_com_runner(
            vectorstore,
            all_documents,
            top_n=args.hybrid_com_top_n,
        )
        pipeline_runners["hybrid_com_rerank"] = build_hybrid_com_rerank_runner(
            vectorstore,
            all_documents,
            top_n=args.hybrid_com_rerank_top_n,
        )

    for pipeline_name in args.pipelines:
        print(f"\n[{pipeline_name}] Building EvaluationDataset...")
        dataset, traces = build_evaluation_dataset(
            rows=rows,
            runner=pipeline_runners[pipeline_name],
            sample_cls=components["SingleTurnSample"],
            dataset_cls=components["EvaluationDataset"],
        )
        save_jsonl(output_dir / f"{pipeline_name}_ragas_inputs.jsonl", traces)

        metrics = make_metrics(
            components,
            include_year_accuracy=args.with_year_accuracy,
        )
        if args.korean_prompts:
            pipeline_logs = adapt_metric_prompts(metrics, evaluator_llm, language="korean")
            prompt_logs.extend([f"{pipeline_name}: {line}" for line in pipeline_logs])

        print(f"[{pipeline_name}] Running ragas.evaluate() on {len(rows)} samples...")
        result = components["evaluate"](
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        df = attach_metadata(result.to_pandas(), rows)
        metric_columns = extract_metric_columns(df)
        all_metric_columns.update(metric_columns)
        per_pipeline_frames[pipeline_name] = df

        csv_path = output_dir / f"{pipeline_name}_ragas_scores.csv"
        df.to_csv(csv_path, index=False)
        print(f"[{pipeline_name}] Saved detailed scores to {csv_path}")

    if per_pipeline_frames:
        save_summary_csv(
            output_dir / "ragas_summary.csv",
            per_pipeline_frames=per_pipeline_frames,
            metric_columns=sorted(all_metric_columns),
            pipeline_order=args.pipelines,
        )
        save_overview_markdown(
            output_dir / "ragas_overview.md",
            per_pipeline_frames=per_pipeline_frames,
            metric_columns=sorted(all_metric_columns),
            args=args,
        )
        save_comparison_csv(
            output_dir / "ragas_comparison.csv",
            per_pipeline_frames=per_pipeline_frames,
            metric_columns=sorted(all_metric_columns),
            pipeline_order=args.pipelines,
        )
        for pipeline_name, df in per_pipeline_frames.items():
            save_readable_pipeline_csv(
                output_dir / f"{pipeline_name}_ragas_readable.csv",
                df=df,
                metric_columns=sorted(all_metric_columns),
            )
        print(f"\nSaved summary to {output_dir / 'ragas_summary.csv'}")

    save_run_metadata(output_dir / "ragas_run_metadata.json", args, evaluator_meta, prompt_logs)
    if prompt_logs:
        print("\nPrompt adaptation log:")
        for log in prompt_logs:
            print(f"- {log}")


if __name__ == "__main__":
    main()
