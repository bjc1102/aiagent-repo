"""
week-2 의료급여 본인부담률 질문 배치 실행 (week-1 DChanHong/V2 패턴).

환경 변수:
  GEMINI_API_KEY       필수
  GEMINI_MODEL         기본 gemini-2.5-flash
  PROMPT_PROFILE       zero_shot | few_shot | chain_of_thought | self_consistency
  RUN_LABEL              결과 폴더 식별용 라벨(선택)
  GEN_TEMPERATURE, GEN_TOP_P, GEN_MAX_TOKENS, GEN_SEED 등 — 호출 시 오버라이드
  EVAL_EXCLUDE_IDS       정답률 계산에서 제외할 id (콤마 구분). Few-shot 예시로 쓴 문항 제외 시 예: q28,q29,q30
  SELF_CONSISTENCY_SAMPLES  1이면 단일 생성, 2 이상이면 다수결 (answer 필드)
  SELF_CONSISTENCY_TEMPERATURES  self_consistency 전용 온도 목록 (예: 0.3,0.5,0.7)
  REFERENCE_MARKDOWN_PATH  기본 week-2/DChanHong/extracted-reference.md
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from prompts import build_system_prompt
from schemas.copayment_response import CopaymentResponse
from services.gemini_service import GeminiService


def _get_env_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value is not None else None


def _get_env_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value is not None else None


def _normalize_answer(text: str) -> str:
    return " ".join(text.strip().split())


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_exclude_ids(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def _parse_env_float_list(name: str) -> list[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _answer_with_retry(
    service: GeminiService,
    question: str,
    *,
    retries: int = 2,
    **run_overrides: object,
) -> tuple[CopaymentResponse, dict[str, int | float | None]]:
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            return service.answer_with_usage(question, **run_overrides)
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    week2_root = base_dir.parent
    data_dir = week2_root / "data"
    dataset_path = data_dir / "dataset.jsonl"
    answer_key_path = data_dir / "answer_key.jsonl"

    prompt_profile = os.getenv("PROMPT_PROFILE", "zero_shot")
    run_label = os.getenv("RUN_LABEL")
    exclude_ids = _parse_exclude_ids(os.getenv("EVAL_EXCLUDE_IDS", ""))
    sc_samples = _get_env_int("SELF_CONSISTENCY_SAMPLES") or 1
    sc_temperatures = _parse_env_float_list("SELF_CONSISTENCY_TEMPERATURES")
    effective_sc_samples = len(sc_temperatures) if sc_temperatures else sc_samples

    generation_overrides = {
        k: v
        for k, v in {
            "temperature": _get_env_float("GEN_TEMPERATURE"),
            "top_p": _get_env_float("GEN_TOP_P"),
            "max_tokens": _get_env_int("GEN_MAX_TOKENS"),
            "presence_penalty": _get_env_float("GEN_PRESENCE_PENALTY"),
            "frequency_penalty": _get_env_float("GEN_FREQUENCY_PENALTY"),
            "seed": _get_env_int("GEN_SEED"),
        }.items()
        if v is not None
    }

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if run_label:
        run_id = f"{run_id}-{run_label}"
    result_dir = base_dir / "json" / "result" / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    service = GeminiService(prompt_profile=prompt_profile)

    reference_md_raw = os.getenv("REFERENCE_MARKDOWN_PATH", "extracted-reference.md")
    reference_md_path = Path(reference_md_raw)
    if not reference_md_path.is_absolute():
        reference_md_path = (base_dir / reference_md_path).resolve()

    reference_source: dict[str, str] | None = None
    if reference_md_path.is_file():
        print(f"📄 참조 마크다운 로드: {reference_md_path}\n")
        extracted_text = reference_md_path.read_text(encoding="utf-8").strip()
        service.system_prompt = build_system_prompt(
            prompt_profile,
            reference_text=extracted_text,
        )
        reference_source = {
            "type": "markdown_file",
            "path": str(reference_md_path),
        }
    else:
        print(
            f"⚠️ REFERENCE_MARKDOWN_PATH 파일 없음 → prompts의 COPAYMENT_REFERENCE 사용: {reference_md_path}\n"
        )

    # 기존 이미지 추출 로직은 보관만 하고 현재 실행에서는 사용하지 않습니다.
    # ref_image_raw = os.getenv("REFERENCE_IMAGE_PATH", "image/image.png")
    # ref_image = Path(ref_image_raw)
    # if not ref_image.is_absolute():
    #     ref_image = (week2_root / ref_image).resolve()
    # extracted_text, ext_meta = service.extract_reference_from_image(ref_image)

    rows = _load_jsonl(dataset_path)
    answer_key = {r["id"]: r["expected_answer"] for r in _load_jsonl(answer_key_path)}

    model_config = service.get_config()
    model_config["run_overrides"] = generation_overrides
    model_config["eval_exclude_ids"] = sorted(exclude_ids)
    model_config["self_consistency_samples"] = effective_sc_samples
    model_config["self_consistency_temperatures"] = sc_temperatures or None
    model_config["reference_markdown_path"] = str(reference_md_path)
    model_config["reference_source"] = reference_source
    if run_label:
        model_config["run_label"] = run_label
    (result_dir / "model-config.json").write_text(
        json.dumps(model_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results: list[dict] = []
    usage_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "elapsed_ms": 0.0,
    }
    requests_with_usage = 0

    correct = 0
    evaluated = 0

    print("🚀 본인부담률 배치 실행\n")

    for row in rows:
        qid = row["id"]
        question = row["question"]
        print(f"📩 ({qid}) {question[:80]}...")

        if effective_sc_samples <= 1:
            parsed, meta = _answer_with_retry(service, question, **generation_overrides)
            answers_for_vote = [parsed.answer]
            final = parsed
        else:
            parsed_list: list[CopaymentResponse] = []
            metas: list[dict] = []
            if prompt_profile == "self_consistency" and sc_temperatures:
                sc_run_temperatures = sc_temperatures
            else:
                base_temperature = generation_overrides.get("temperature")
                sc_run_temperatures = [base_temperature] * effective_sc_samples

            for temp in sc_run_temperatures:
                run_overrides = dict(generation_overrides)
                if temp is not None:
                    run_overrides["temperature"] = temp
                one, meta = _answer_with_retry(service, question, **run_overrides)
                parsed_list.append(one)
                metas.append(meta)
            votes = [p.answer for p in parsed_list]
            counted = Counter(_normalize_answer(a) for a in votes)
            winner, win_count = counted.most_common(1)[0]
            display_answer = next(a for a in votes if _normalize_answer(a) == winner)
            reason_pick = next(
                p.reason for p in parsed_list if _normalize_answer(p.answer) == winner
            )
            parsed = CopaymentResponse(
                answer=display_answer,
                reason=f"[Self-Consistency {win_count}/{effective_sc_samples}] {reason_pick}",
            )
            meta = {
                "prompt_tokens": sum(m.get("prompt_tokens") or 0 for m in metas),
                "completion_tokens": sum(m.get("completion_tokens") or 0 for m in metas),
                "total_tokens": sum(m.get("total_tokens") or 0 for m in metas),
                "elapsed_ms": sum(float(m.get("elapsed_ms") or 0) for m in metas),
                "self_consistency_votes": votes,
                "self_consistency_temperatures": sc_run_temperatures,
            }

        expected = answer_key.get(qid)
        norm_pred = _normalize_answer(parsed.answer)
        norm_exp = _normalize_answer(expected) if expected is not None else None
        is_correct = norm_exp is not None and norm_pred == norm_exp
        excluded = qid in exclude_ids

        if not excluded and expected is not None:
            evaluated += 1
            if is_correct:
                correct += 1

        if any(meta.get(k) is not None for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            requests_with_usage += 1
            usage_totals["prompt_tokens"] += meta.get("prompt_tokens") or 0
            usage_totals["completion_tokens"] += meta.get("completion_tokens") or 0
            usage_totals["total_tokens"] += meta.get("total_tokens") or 0
        usage_totals["elapsed_ms"] += float(meta.get("elapsed_ms") or 0)

        results.append(
            {
                "id": qid,
                "difficulty": row.get("difficulty"),
                "question": question,
                "expected_answer": expected,
                "predicted": parsed.model_dump(),
                "match": is_correct,
                "excluded_from_eval": excluded,
                "response_metadata": meta,
            }
        )

        print(f"   → answer: {parsed.answer!r}  (정답: {expected!r}, 일치: {is_correct}, 평가제외: {excluded})")
        print("-" * 50)

    summary = {
        "model": service.model_name,
        "prompt_profile": prompt_profile,
        "total_items": len(rows),
        "evaluated_items": evaluated,
        "correct": correct,
        "accuracy": round(correct / evaluated, 4) if evaluated else None,
        "eval_exclude_ids": sorted(exclude_ids),
        "self_consistency_samples": effective_sc_samples,
    }

    (result_dir / "analysis-results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (result_dir / "eval-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    usage_summary = {
        "model": service.model_name,
        "request_count": len(rows),
        "self_consistency_multiplier": max(1, effective_sc_samples),
        "requests_with_usage": requests_with_usage,
        "totals": usage_totals,
        "averages_per_request": {
            "prompt_tokens": round(usage_totals["prompt_tokens"] / requests_with_usage, 2)
            if requests_with_usage
            else 0,
            "completion_tokens": round(usage_totals["completion_tokens"] / requests_with_usage, 2)
            if requests_with_usage
            else 0,
            "total_tokens": round(usage_totals["total_tokens"] / requests_with_usage, 2)
            if requests_with_usage
            else 0,
            "elapsed_ms": round(usage_totals["elapsed_ms"] / len(rows), 2) if rows else 0,
        },
    }
    (result_dir / "usage-summary.json").write_text(
        json.dumps(usage_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n📊 요약:", json.dumps(summary, ensure_ascii=False))
    print(f"결과 디렉터리: {result_dir}")


if __name__ == "__main__":
    main()
