import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from prompts import (
    get_step1_prompt,
    get_step2_prompt,
    get_step3_prompt,
    get_step4_prompt,
    get_step5_prompt,
    get_step6_prompt,
)

# =========================================================
# 0. 기본 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "gpt-5.4-mini"
DATASET_FILE = BASE_DIR / "dataset.jsonl"
ANSWER_KEY_FILE = BASE_DIR / "answer_key.jsonl"
RESULTS_DIR = BASE_DIR / "results"

# 실행할 step 선택
ENABLED_STEPS = [
    # "zero-shot Baseline",
    # "few-shot Prompting",
    # "chain-of-thought Prompting",
    # "self-consistency",
    # "few-shot-cot",
    "json-few-shot-cot-self-consistency",
]

EXCLUDE_IDS_MAP = {
    "zero-shot Baseline": [],
    "few-shot Prompting": [],
    "chain-of-thought Prompting": [],
    "self-consistency": [],
    "few-shot-cot": [],
    "json-few-shot-cot-self-consistency": [],
}

SELF_CONSISTENCY_RUNS = 5
SELF_CONSISTENCY_TEMPERATURE = 0.7

STEP_CONFIG = {
    "zero-shot Baseline": {
        "slug": "zero_shot_baseline",
        "prompt_func": get_step1_prompt,
        "temperature": 0.0,
        "runs": 1,
        "schema_type": "reason",
    },
    "few-shot Prompting": {
        "slug": "few_shot_prompting",
        "prompt_func": get_step2_prompt,
        "temperature": 0.0,
        "runs": 1,
        "schema_type": "reason",
    },
    "chain-of-thought Prompting": {
        "slug": "chain_of_thought_prompting",
        "prompt_func": get_step3_prompt,
        "temperature": 0.0,
        "runs": 1,
        "schema_type": "reason_steps",
    },
    "self-consistency": {
        "slug": "self_consistency",
        "prompt_func": get_step4_prompt,
        "temperature": SELF_CONSISTENCY_TEMPERATURE,
        "runs": SELF_CONSISTENCY_RUNS,
        "schema_type": "reason_steps",
    },
    "few-shot-cot": {
        "slug": "few_shot_cot",
        "prompt_func": get_step5_prompt,
        "temperature": 0.0,
        "runs": 1,
        "schema_type": "reason_steps",
    },
    "json-few-shot-cot-self-consistency": {
        "slug": "json_few_shot_cot_self_consistency",
        "prompt_func": get_step6_prompt,
        "temperature": SELF_CONSISTENCY_TEMPERATURE,
        "runs": SELF_CONSISTENCY_RUNS,
        "schema_type": "reason_steps",
    },
}

# =========================================================
# 1. 환경 준비
# =========================================================

load_dotenv(BASE_DIR / ".env")
client = OpenAI()
RESULTS_DIR.mkdir(exist_ok=True)

# =========================================================
# 2. 데이터 로드
# =========================================================

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_answer_map(answer_key: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["id"]: row for row in answer_key}


# =========================================================
# 3. 정답 정규화
# =========================================================

def normalize_answer(text: str) -> str:
    if text is None:
        return ""

    value = str(text).strip().replace(" ", "")

    mapping = {
        "없음": "무료",
        "없음(무료)": "무료",
        "면제": "무료",
        "0원": "무료",
        "1000원": "1,000원",
        "1500원": "1,500원",
        "2000원": "2,000원",
        "해당없음": "해당되지않음",
    }

    value = mapping.get(value, value)
    value = value.replace("/", ",")
    return value


# =========================================================
# 4. 모델 호출
# =========================================================

def call_model_once(question: str, prompt_text: str, step_name: str) -> Dict[str, Any]:
    config = STEP_CONFIG[step_name]

    response = client.responses.create(
        model=MODEL_NAME,
        temperature=config["temperature"],
        input=[
            {"role": "developer", "content": prompt_text},
            {"role": "user", "content": question},
        ],
    )

    return json.loads(response.output_text)


def call_model_self_consistency(question: str, prompt_text: str, step_name: str) -> Dict[str, Any]:
    config = STEP_CONFIG[step_name]

    outputs = []
    for _ in range(config["runs"]):
        outputs.append(call_model_once(question, prompt_text, step_name))

    answers = [normalize_answer(o.get("answer", "")) for o in outputs if o.get("answer", "")]
    if not answers:
        raise ValueError("Self-consistency answer 추출 실패")

    final_answer = Counter(answers).most_common(1)[0][0]
    representative = next(
        (o for o in outputs if normalize_answer(o.get("answer", "")) == final_answer),
        outputs[0],
    )

    return {
        "answer": final_answer,
        "reason_steps": representative.get("reason_steps", []),
        "all_answers": answers,
        "raw_outputs": outputs,
    }


# =========================================================
# 5. Step 실행
# =========================================================

def run_step(
    step_name: str,
    prompt_text: str,
    dataset: List[Dict[str, Any]],
    answer_map: Dict[str, Dict[str, Any]],
    exclude_ids: List[str],
) -> Dict[str, Any]:
    config = STEP_CONFIG[step_name]

    total = 0
    correct = 0
    results = []
    difficulty_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for item in dataset:
        qid = item["id"]

        if qid in exclude_ids:
            continue

        question = item["question"]
        difficulty = item.get("difficulty", "unknown")
        expected = normalize_answer(answer_map[qid]["expected_answer"])

        try:
            if config["runs"] > 1:
                pred = call_model_self_consistency(question, prompt_text, step_name)
            else:
                pred = call_model_once(question, prompt_text, step_name)

            pred_answer = normalize_answer(pred.get("answer", ""))
            is_correct = pred_answer == expected

            if "reason" in pred:
                model_reason = pred.get("reason", "")
            else:
                model_reason = " | ".join(pred.get("reason_steps", []))

        except Exception as e:
            pred = {}
            pred_answer = "ERROR"
            model_reason = f"모델 호출 실패: {repr(e)}"
            is_correct = False
            print(f"[ERROR] {step_name} | {qid} | {repr(e)}")

        total += 1
        difficulty_stats[difficulty]["total"] += 1

        if is_correct:
            correct += 1
            difficulty_stats[difficulty]["correct"] += 1

        row = {
            "id": qid,
            "question": question,
            "difficulty": difficulty,
            "expected": expected,
            "pred": pred_answer,
            "model_reason": model_reason,
            "correct": is_correct,
        }

        if config["runs"] > 1:
            row["all_answers"] = pred.get("all_answers", [])
            row["raw_outputs"] = pred.get("raw_outputs", [])

        results.append(row)

        print(f"[{step_name}] {qid} → {pred_answer} ({'정답' if is_correct else '오답'})")

    accuracy = round(correct / total * 100, 2) if total > 0 else 0.0

    return {
        "step": step_name,
        "slug": config["slug"],
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "difficulty_stats": dict(difficulty_stats),
        "results": results,
    }


# =========================================================
# 6. 파일 저장
# =========================================================

def save_json(file_path: Path, data: Dict[str, Any]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_prompt_text(step_name: str, prompt_text: str) -> None:
    slug = STEP_CONFIG[step_name]["slug"]
    path = RESULTS_DIR / f"{slug}_prompt.txt"
    path.write_text(prompt_text, encoding="utf-8")


def make_markdown_summary(all_results: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("# 실험 결과 요약")
    lines.append("")
    lines.append(f"- 사용 모델: `{MODEL_NAME}`")
    lines.append("")
    lines.append("## Step별 전체 정답률")
    lines.append("")
    lines.append("| Step | 전체 문항 수 | 정답 수 | 오답 수 | 정답률 |")
    lines.append("|---|---:|---:|---:|---:|")

    for result in all_results:
        lines.append(
            f"| {result['step']} | {result['total']} | {result['correct']} | {result['wrong']} | {result['accuracy']}% |"
        )

    lines.append("")

    for result in all_results:
        step_name = result["step"]
        config = STEP_CONFIG[step_name]

        lines.append(f"## {step_name}")
        lines.append("")
        lines.append(f"- 전체 문항 수: {result['total']}")
        lines.append(f"- 정답 수: {result['correct']}")
        lines.append(f"- 오답 수: {result['wrong']}")
        lines.append(f"- 정답률: {result['accuracy']}%")
        lines.append("")

        if config["runs"] > 1:
            lines.append(f"- 반복 횟수: {config['runs']}")
            lines.append(f"- temperature: {config['temperature']}")
            lines.append("")

        lines.append("### 난이도별 결과")
        lines.append("")
        lines.append("| 난이도 | 전체 | 정답 | 정답률 |")
        lines.append("|---|---:|---:|---:|")

        for difficulty, stats in result["difficulty_stats"].items():
            diff_total = stats["total"]
            diff_correct = stats["correct"]
            diff_acc = round((diff_correct / diff_total * 100), 2) if diff_total > 0 else 0.0
            lines.append(f"| {difficulty} | {diff_total} | {diff_correct} | {diff_acc}% |")

        lines.append("")
        lines.append("### 오답 목록")
        lines.append("")

        wrong_rows = [row for row in result["results"] if not row["correct"]]

        if not wrong_rows:
            lines.append("- 없음")
        else:
            for row in wrong_rows:
                if config["runs"] > 1:
                    lines.append(
                        f"- {row['id']} | 예측: {row['pred']} | 정답: {row['expected']} | 전체 응답: {row.get('all_answers', [])} | 질문: {row['question']}"
                    )
                else:
                    lines.append(
                        f"- {row['id']} | 예측: {row['pred']} | 정답: {row['expected']} | 질문: {row['question']}"
                    )

        lines.append("")

    return "\n".join(lines)


# =========================================================
# 7. 메인 실행
# =========================================================

def main():
    dataset = load_jsonl(DATASET_FILE)
    answer_key = load_jsonl(ANSWER_KEY_FILE)
    answer_map = build_answer_map(answer_key)

    all_results = []

    print("=" * 70)
    print("실험 시작")
    print("=" * 70)
    print(f"모델: {MODEL_NAME}")
    print(f"실행 대상 Step: {', '.join(ENABLED_STEPS)}")
    print(f"문항 수: {len(dataset)}")
    print()

    for step_name in ENABLED_STEPS:
        print(f"\n===== {step_name} =====")

        prompt = STEP_CONFIG[step_name]["prompt_func"]()
        exclude_ids = EXCLUDE_IDS_MAP.get(step_name, [])

        save_prompt_text(step_name, prompt)

        result = run_step(step_name, prompt, dataset, answer_map, exclude_ids)
        all_results.append(result)

        save_path = RESULTS_DIR / f"{STEP_CONFIG[step_name]['slug']}.json"
        save_json(save_path, result)

    summary_md = make_markdown_summary(all_results)
    summary_path = RESULTS_DIR / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    print("\n===== 전체 결과 =====")
    for r in all_results:
        print(f"{r['step']} → {r['accuracy']}%")

    print(f"\nsummary.md 저장 완료: {summary_path}")


if __name__ == "__main__":
    main()