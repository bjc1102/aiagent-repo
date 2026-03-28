"""
2주차 과제: 프롬프트 엔지니어링으로 의료급여 본인부담률 정답률 개선
- Step 1: Zero-shot
- Step 2: Few-shot
- Step 3: Chain-of-Thought (CoT)
- Step 4: Self-Consistency (CoT + 다수결)
- Step 5: (선택) 추가 실험
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# ── 설정 ──────────────────────────────────────────────
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

client = Anthropic()

# ── Pydantic 스키마 ───────────────────────────────────

class Answer(BaseModel):
    answer: str
    reason: str


class CoTAnswer(BaseModel):
    reasoning_steps: list[str]
    answer: str
    reason: str


# ── 의료급여 본인부담률 참조 데이터 ─────────────────────

COPAYMENT_REFERENCE = """
## 의료급여 본인부담률 참조 데이터

### 01. 외래 및 입원 시 본인부담률 (기본)

| 구분 | 1종 수급권자 | 2종 수급권자 |
|------|------------|------------|
| 입원 | 무료 | 10% |
| 외래 - 제1차의료급여기관 | 1,000원 | 1,000원 |
| 외래 - 제2차의료급여기관 | 1,500원 | 15% |
| 외래 - 제3차의료급여기관 | 2,000원 | 15% |
| 약국 | 500원 | 500원 |

### 02. 15세 이하 아동

**입원:**
| 나이 | 본인부담률 |
|------|----------|
| 6세 미만 (1세 이상) | 무료 |
| 6세 이상 ~ 15세 이하 | 3% |

**외래:**
| 나이 | 제1차의료급여기관 | 제2·3차의료급여기관 |
|------|-----------------|-------------------|
| 1세 미만 | 무료 | 5% |
| 1세 미만 만성질환자 | 무료 | 무료 |
| 1세 이상 ~ 6세 미만 | 무료 | 5% |

### 03. 분만 및 임신부

**입원:**
| 구분 | 본인부담률 |
|------|----------|
| 자연분만 | 무료 |
| 제왕절개 | 무료 |
| 유산·사산 | 무료 |
| 고위험 임신부 | 5% |

### 04. 65세 이상 틀니·임플란트

| 구분 | 1종 | 2종 |
|------|-----|-----|
| 틀니 | 5% | 15% |
| 임플란트 | 10% | 20% |

※ 본인부담 보상제·상한제 해당되지 않음

### 05. 추나요법

| 구분 | 1종 | 2종 |
|------|-----|-----|
| 디스크·협착증 - 복잡추나 | 30% | 50% |
| 디스크·협착증 - 단순추나 | 30% | 50% |
| 디스크·협착증 외 - 복잡추나 | 50% | 80% |
| 디스크·협착증 외 - 단순추나 | 50% | 80% |

### 06. 치아 홈메우기

**입원:**
| 나이 | 본인부담률 |
|------|----------|
| 6세 이상 ~ 15세 이하 | 3% |
| 16세 이상 ~ 18세 이하 | 5% |

### 07. 정신질환 외래진료 (2종 수급권자)

| 구분 | 본인부담률 |
|------|----------|
| 조현병 | 병원급 이상 5% |
| 치매질환 입원 및 병원급 이상 외래진료 | 5% |
| 조현병 외 정신질환 | 병원급 이상 10% |

### 08. CT, MRI, PET (2종 수급권자)

| 대상 | 제1차의료급여기관 | 제2·3차의료급여기관 |
|------|-----------------|-------------------|
| 일반 | 5% | 15% |
| 조현병 등 정신질환자 | 5% | 15% |
| 임신부 (유산·사산 포함) | 5% | 5% |
| 1세 미만 만성질환자 | 5% | 5% |
"""


# ── 프롬프트 정의 ─────────────────────────────────────

SYSTEM_ZERO_SHOT = f"""아래는 의료급여 본인부담률 참조 데이터입니다.
질문에 대해 정확한 본인부담률을 답하세요. 답만 간결하게 작성하세요.

{COPAYMENT_REFERENCE}

반드시 아래 JSON 형식으로만 응답하세요:
{{"answer": "답변", "reason": "근거"}}
"""

FEW_SHOT_EXAMPLES = """
다음은 질문-답변 예시입니다:

예시 1)
Q: 1종 수급권자가 입원하면 본인부담률은?
A: {{"answer": "무료", "reason": "01번 표 → 1종 → 입원 = 무료"}}

예시 2)
Q: 2종 수급권자인 3세 아동이 입원하면 본인부담률은?
A: {{"answer": "무료", "reason": "02번 15세 이하 아동 → 입원 → 6세 미만 = 무료"}}

예시 3)
Q: 2종 수급권자인 조현병 환자가 외래 진료를 받으면 본인부담률은?
A: {{"answer": "병원급 이상 5%", "reason": "07번 정신질환 외래진료 → 조현병 = 병원급 이상 5%"}}
"""

SYSTEM_FEW_SHOT = f"""아래는 의료급여 본인부담률 참조 데이터입니다.
질문에 대해 정확한 본인부담률을 답하세요.

{COPAYMENT_REFERENCE}

{FEW_SHOT_EXAMPLES}

반드시 아래 JSON 형식으로만 응답하세요:
{{"answer": "답변", "reason": "근거"}}
"""

SYSTEM_COT = f"""아래는 의료급여 본인부담률 참조 데이터입니다.
질문에 대해 정확한 본인부담률을 답하세요.

{COPAYMENT_REFERENCE}

{FEW_SHOT_EXAMPLES}

답변 전에 반드시 아래 단계를 따라 추론하세요:
1. 수급권자 종별 (1종/2종) 확인
2. 해당되는 특수 조건 확인 (나이, 질환, 임신 등)
3. 적용할 표 번호 결정 (01~08번)
4. 해당 표에서 정확한 본인부담률 찾기
5. 금액 계산이 필요한 경우 계산 수행

반드시 아래 JSON 형식으로만 응답하세요:
{{"reasoning_steps": ["단계1", "단계2", ...], "answer": "답변", "reason": "근거"}}
"""

# Few-shot에서 사용한 예시의 원본 질문과 겹치지 않도록 제외할 ID는 없음
# (예시는 dataset에 없는 별도 질문으로 구성)
FEW_SHOT_EXCLUDE_IDS: set[str] = set()


# ── 데이터 로드 ───────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_data():
    questions = load_jsonl(DATA_DIR / "dataset.jsonl")
    answers = {a["id"]: a for a in load_jsonl(DATA_DIR / "answer_key.jsonl")}
    return questions, answers


# ── LLM 호출 ─────────────────────────────────────────

def call_llm(system: str, question: str, temperature: float = 0.0) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def parse_json_response(text: str) -> dict:
    """LLM 응답에서 JSON을 추출하여 파싱"""
    # ```json ... ``` 블록이 있으면 추출
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    # { ... } 블록 추출
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


# ── 정답 비교 ─────────────────────────────────────────

def normalize_answer(ans: str) -> str:
    """정답 비교를 위한 정규화"""
    ans = ans.strip().replace(" ", "").replace(",", "")
    # "원" 단위 통일
    ans = ans.replace("원", "")
    return ans


def is_correct(predicted: str, expected: str) -> bool:
    """정답 여부 판별 (정규화 후 비교)"""
    pred = normalize_answer(predicted)
    exp = normalize_answer(expected)
    # 정확히 일치
    if pred == exp:
        return True
    # expected가 pred에 포함
    if exp in pred or pred in exp:
        return True
    return False


# ── Step 1: Zero-shot ─────────────────────────────────

def run_zero_shot(questions: list[dict], answers: dict) -> dict:
    print("\n" + "=" * 60)
    print("Step 1: Zero-shot Baseline")
    print("=" * 60)

    results = []
    correct = 0

    for q in questions:
        qid = q["id"]
        if qid in FEW_SHOT_EXCLUDE_IDS:
            continue
        try:
            raw = call_llm(SYSTEM_ZERO_SHOT, q["question"])
            parsed = parse_json_response(raw)
            predicted = parsed.get("answer", "")
            expected = answers[qid]["expected_answer"]
            match = is_correct(predicted, expected)
            if match:
                correct += 1
            results.append({
                "id": qid,
                "difficulty": q["difficulty"],
                "predicted": predicted,
                "expected": expected,
                "correct": match,
                "reason": parsed.get("reason", ""),
            })
            status = "O" if match else "X"
            print(f"  [{status}] {qid} ({q['difficulty']}): {predicted} (정답: {expected})")
        except Exception as e:
            print(f"  [E] {qid}: {e}")
            results.append({
                "id": qid, "difficulty": q["difficulty"],
                "predicted": "ERROR", "expected": answers[qid]["expected_answer"],
                "correct": False, "reason": str(e),
            })

    total = len(results)
    accuracy = correct / total * 100 if total else 0
    print(f"\n  정답률: {correct}/{total} ({accuracy:.1f}%)")

    return {"step": "zero_shot", "accuracy": accuracy, "correct": correct,
            "total": total, "results": results}


# ── Step 2: Few-shot ──────────────────────────────────

def run_few_shot(questions: list[dict], answers: dict) -> dict:
    print("\n" + "=" * 60)
    print("Step 2: Few-shot Prompting")
    print("=" * 60)

    results = []
    correct = 0

    for q in questions:
        qid = q["id"]
        if qid in FEW_SHOT_EXCLUDE_IDS:
            continue
        try:
            raw = call_llm(SYSTEM_FEW_SHOT, q["question"])
            parsed = parse_json_response(raw)
            predicted = parsed.get("answer", "")
            expected = answers[qid]["expected_answer"]
            match = is_correct(predicted, expected)
            if match:
                correct += 1
            results.append({
                "id": qid, "difficulty": q["difficulty"],
                "predicted": predicted, "expected": expected,
                "correct": match, "reason": parsed.get("reason", ""),
            })
            status = "O" if match else "X"
            print(f"  [{status}] {qid} ({q['difficulty']}): {predicted} (정답: {expected})")
        except Exception as e:
            print(f"  [E] {qid}: {e}")
            results.append({
                "id": qid, "difficulty": q["difficulty"],
                "predicted": "ERROR", "expected": answers[qid]["expected_answer"],
                "correct": False, "reason": str(e),
            })

    total = len(results)
    accuracy = correct / total * 100 if total else 0
    print(f"\n  정답률: {correct}/{total} ({accuracy:.1f}%)")

    return {"step": "few_shot", "accuracy": accuracy, "correct": correct,
            "total": total, "results": results}


# ── Step 3: Chain-of-Thought ──────────────────────────

def run_cot(questions: list[dict], answers: dict) -> dict:
    print("\n" + "=" * 60)
    print("Step 3: Chain-of-Thought (CoT)")
    print("=" * 60)

    results = []
    correct = 0

    for q in questions:
        qid = q["id"]
        if qid in FEW_SHOT_EXCLUDE_IDS:
            continue
        try:
            raw = call_llm(SYSTEM_COT, q["question"])
            parsed = parse_json_response(raw)
            predicted = parsed.get("answer", "")
            expected = answers[qid]["expected_answer"]
            match = is_correct(predicted, expected)
            if match:
                correct += 1
            results.append({
                "id": qid, "difficulty": q["difficulty"],
                "predicted": predicted, "expected": expected,
                "correct": match, "reason": parsed.get("reason", ""),
                "reasoning_steps": parsed.get("reasoning_steps", []),
            })
            status = "O" if match else "X"
            print(f"  [{status}] {qid} ({q['difficulty']}): {predicted} (정답: {expected})")
        except Exception as e:
            print(f"  [E] {qid}: {e}")
            results.append({
                "id": qid, "difficulty": q["difficulty"],
                "predicted": "ERROR", "expected": answers[qid]["expected_answer"],
                "correct": False, "reason": str(e),
            })

    total = len(results)
    accuracy = correct / total * 100 if total else 0
    print(f"\n  정답률: {correct}/{total} ({accuracy:.1f}%)")

    return {"step": "cot", "accuracy": accuracy, "correct": correct,
            "total": total, "results": results}


# ── Step 4: Self-Consistency ──────────────────────────

def run_self_consistency(questions: list[dict], answers: dict,
                         n_samples: int = 5, temperature: float = 0.7) -> dict:
    print("\n" + "=" * 60)
    print(f"Step 4: Self-Consistency (n={n_samples}, temp={temperature})")
    print("=" * 60)

    results = []
    correct = 0

    for q in questions:
        qid = q["id"]
        if qid in FEW_SHOT_EXCLUDE_IDS:
            continue
        try:
            # 여러 번 생성
            sampled_answers = []
            for _ in range(n_samples):
                raw = call_llm(SYSTEM_COT, q["question"], temperature=temperature)
                parsed = parse_json_response(raw)
                sampled_answers.append(parsed.get("answer", ""))

            # 다수결 투표 (정규화 후)
            normalized = [normalize_answer(a) for a in sampled_answers]
            counter = Counter(normalized)
            majority_normalized = counter.most_common(1)[0][0]

            # 원본 답변 중 다수결과 일치하는 첫 번째 것 사용
            predicted = majority_normalized
            for sa in sampled_answers:
                if normalize_answer(sa) == majority_normalized:
                    predicted = sa
                    break

            expected = answers[qid]["expected_answer"]
            match = is_correct(predicted, expected)
            if match:
                correct += 1

            results.append({
                "id": qid, "difficulty": q["difficulty"],
                "predicted": predicted, "expected": expected,
                "correct": match,
                "all_samples": sampled_answers,
                "vote_counts": dict(counter),
            })
            status = "O" if match else "X"
            print(f"  [{status}] {qid} ({q['difficulty']}): {predicted} "
                  f"(정답: {expected}) [투표: {dict(counter)}]")
        except Exception as e:
            print(f"  [E] {qid}: {e}")
            results.append({
                "id": qid, "difficulty": q["difficulty"],
                "predicted": "ERROR", "expected": answers[qid]["expected_answer"],
                "correct": False, "reason": str(e),
            })

    total = len(results)
    accuracy = correct / total * 100 if total else 0
    print(f"\n  정답률: {correct}/{total} ({accuracy:.1f}%)")

    return {"step": "self_consistency", "accuracy": accuracy, "correct": correct,
            "total": total, "n_samples": n_samples, "temperature": temperature,
            "results": results}


# ── 결과 저장 ─────────────────────────────────────────

def save_results(all_results: list[dict], filename: str = "results.json"):
    output_path = Path(__file__).resolve().parent / filename
    summary = {
        "model": MODEL,
        "steps": []
    }
    for r in all_results:
        step_summary = {
            "step": r["step"],
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "total": r["total"],
        }
        if "n_samples" in r:
            step_summary["n_samples"] = r["n_samples"]
            step_summary["temperature"] = r["temperature"]

        # 난이도별 정답률
        by_diff = {}
        for item in r["results"]:
            d = item["difficulty"]
            if d not in by_diff:
                by_diff[d] = {"correct": 0, "total": 0}
            by_diff[d]["total"] += 1
            if item["correct"]:
                by_diff[d]["correct"] += 1
        for d in by_diff:
            by_diff[d]["accuracy"] = (
                by_diff[d]["correct"] / by_diff[d]["total"] * 100
                if by_diff[d]["total"] else 0
            )
        step_summary["by_difficulty"] = by_diff
        step_summary["results"] = r["results"]
        summary["steps"].append(step_summary)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장 완료: {output_path}")


# ── 메인 ─────────────────────────────────────────────

def main():
    questions, answers = load_data()

    all_results = []

    # Step 1: Zero-shot
    r1 = run_zero_shot(questions, answers)
    all_results.append(r1)

    # Step 2: Few-shot
    r2 = run_few_shot(questions, answers)
    all_results.append(r2)

    # Step 3: CoT
    r3 = run_cot(questions, answers)
    all_results.append(r3)

    # Step 4: Self-Consistency
    r4 = run_self_consistency(questions, answers, n_samples=5, temperature=0.7)
    all_results.append(r4)

    # 결과 저장
    save_results(all_results)

    # 최종 요약
    print("\n" + "=" * 60)
    print("최종 요약")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['step']:20s}: {r['correct']:2d}/{r['total']:2d} ({r['accuracy']:.1f}%)")


if __name__ == "__main__":
    main()
