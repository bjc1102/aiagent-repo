import json
from pathlib import Path
from typing import List, Dict
from collections import Counter
from pydantic import BaseModel
from google import genai
from google.genai import types


# ---------------------------
# 1. 설정
# ---------------------------
MODEL_NAME = "gemini-2.5-flash"
N_SAMPLES = 5
TEMPERATURE = 0.7


# ---------------------------
# 2. 응답 스키마
# ---------------------------
class CopaymentCoTResult(BaseModel):
    answer: str
    reasoning_steps: List[str]
    reason: str


# ---------------------------
# 3. JSONL 로드
# ---------------------------
def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


dataset = load_jsonl("../../data/dataset.jsonl")
answer_key = load_jsonl("../../data/answer_key.jsonl")
answer_map = {row["id"]: row for row in answer_key}


# ---------------------------
# 4. 참조 데이터
# ---------------------------
copayment_reference = """
[규칙]

1. 65세 이상 + 1종 수급권자 + 틀니: 본인부담률 5%
2. 65세 이상 + 2종 수급권자 + 틀니: 본인부담률 15%
3. 65세 이상 + 1종 수급권자 + 치과 임플란트: 본인부담률 10%
4. 65세 이상 + 2종 수급권자 + 치과 임플란트: 본인부담률 20%

5. 추나요법 + 복잡추나 + 디스크/협착증 + 1종: 본인부담률 30%
6. 추나요법 + 복잡추나 + 디스크/협착증 + 2종: 본인부담률 40%
7. 추나요법 + 복잡추나 + 디스크/협착증 외 + 1종: 본인부담률 80%
8. 추나요법 + 복잡추나 + 디스크/협착증 외 + 2종: 본인부담률 80%

9. 추나요법 + 단순추나/특수추나 + 디스크/협착증 + 1종: 본인부담률 30%
10. 추나요법 + 단순추나/특수추나 + 디스크/협착증 + 2종: 본인부담률 40%
11. 추나요법 + 단순추나/특수추나 + 디스크/협착증 외 + 1종: 본인부담률 30%
12. 추나요법 + 단순추나/특수추나 + 디스크/협착증 외 + 2종: 본인부담률 40%

13. 의료급여 2종 수급권자 + 치아 홈메우기 + 입원 + 16세 이상 ~ 18세 이하: 본인부담률 5%
14. 의료급여 2종 수급권자 + 치아 홈메우기 + 입원 + 6세 이상 ~ 15세 이하: 본인부담률 3%
15. 의료급여 2종 수급권자 + 치아 홈메우기 + 입원 + 6세 미만: 본인부담 없음
16. 의료급여 2종 수급권자 + 치아 홈메우기 + 외래 + 18세 이하 + 병원급 이상: 본인부담률 5%

17. 의료급여 2종 수급권자 + 분만 + 입원 + 자연분만: 본인부담 없음
18. 의료급여 2종 수급권자 + 분만 + 입원 + 제왕절개분만: 본인부담 없음
19. 의료급여 2종 수급권자 + 임신부 + 입원 + 고위험 임신부: 본인부담률 5%
20. 의료급여 2종 수급권자 + 임신부(유산·사산 포함) + 외래 + 병원급 이상: 본인부담률 5%

21. 15세 이하 아동 + 입원 + 6세 미만: 본인부담 없음
22. 15세 이하 아동 + 입원 + 6세 이상 ~ 15세 이하: 본인부담률 3%
23. 15세 이하 아동 + 외래 + 1세 미만 + 제1차의료급여기관: 본인부담 없음
24. 15세 이하 아동 + 외래 + 1세 미만 + 제2·3차의료급여기관: 본인부담률 5%
25. 15세 이하 아동 + 외래 + 1세 미만 만성질환자 + 제2차의료급여기관: 본인부담 없음
26. 15세 이하 아동 + 외래 + 5세까지의 조산아·저체중출생아 + 병원급 이상: 본인부담률 5%

27. 정신질환 외래진료 + 조현병 + 병원급 이상: 본인부담률 5%
28. 정신질환 외래진료 + 조현병 외 정신질환 + 병원급 이상: 본인부담률 10%
29. 정신질환 외래진료 + 장기지속형 주사제: 본인부담률 5%

30. 치매질환 + 입원: 본인부담률 5%
31. 치매질환 + 외래 + 병원급 이상: 본인부담률 5%

32. CT/MRI/PET 등 + 임신부(유산·사산 포함) + 제1차의료급여기관: 본인부담률 5%
33. CT/MRI/PET 등 + 5세까지의 조산아 및 저체중 출생아 + 제1차의료급여기관: 본인부담률 5%
34. CT/MRI/PET 등 + 치매질환자 + 제1차의료급여기관: 본인부담률 5%
35. CT/MRI/PET 등 + 1세 미만 만성질환자 + 제2차의료급여기관: 본인부담률 5%
36. CT/MRI/PET 등 + 조현병 등 정신질환자 + 제2·3차의료급여기관: 본인부담률 15%

[비고]
- 65세 이상 틀니/치과 임플란트 항목은 본인부담 보상제·상한제에 해당되지 않음
- 65세 이상 틀니/치과 임플란트 항목에서 2종 장애인의 경우 장애인 의료비 지원 없음
- 정신질환 외래진료의 장기지속형 주사제 5%는 1종·2종 모두 해당하며, 외래본인부담면제자는 제외
""".strip()


# ---------------------------
# 5. 영어 시스템 프롬프트
# ---------------------------
SYSTEM_PROMPT_COT_EN = f"""
You are a medical copayment decision assistant for the Korean Medical Aid system.

You must answer ONLY based on the reference data below.
Do NOT make assumptions outside the given rules.
Return your answer strictly in JSON format.

Output format:
{{
  "answer": "...",
  "reasoning_steps": [
    "step 1",
    "step 2",
    "step 3"
  ],
  "reason": "final applied rule"
}}

Decision process:
1. Identify the beneficiary type (Type 1 or Type 2).
2. Identify the age condition.
3. Identify the disease or special condition.
4. Identify whether the case is inpatient or outpatient.
5. Identify the medical institution type if mentioned.
6. Find the MOST specific matching rule.
7. Write only the final copayment in the answer field.

Important rules:
- More specific rules override more general rules.
- Keep reasoning_steps short and step-by-step.
- Keep reason as a one-line summary of the final applied rule.
- The answer field must contain only the final copayment result.
- If the result is no copayment, output "무료" to match the evaluation format.

[REFERENCE DATA]
{copayment_reference}
""".strip()


# ---------------------------
# 6. 정답 정규화
# ---------------------------
def normalize_answer(text: str) -> str:
    if text is None:
        return ""

    text = text.strip()
    text = text.replace("％", "%")
    text = text.replace("퍼센트", "%")
    text = text.replace("본인부담 없음", "무료")
    text = text.replace("본인부담률", "")
    text = text.replace("본인부담금", "")
    text = text.replace(",", "")
    text = text.replace("없음", "무료")
    text = text.replace(" ", "")
    text = text.replace("적용되지 않습니다.", "해당되지 않음")
    return text


# ---------------------------
# 7. Gemini 클라이언트
# ---------------------------
client = genai.Client()


# ---------------------------
# 8. 단일 영어 CoT 응답 생성
# ---------------------------
def generate_single_cot_en(question: str) -> CopaymentCoTResult:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            SYSTEM_PROMPT_COT_EN,
            question,
        ],
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            response_mime_type="application/json",
            response_schema=CopaymentCoTResult,
        ),
    )

    if hasattr(response, "parsed") and response.parsed is not None:
        return response.parsed

    data = json.loads(response.text)
    return CopaymentCoTResult(**data)


# ---------------------------
# 9. Self-Consistency
# ---------------------------
def solve_self_consistency_en(question: str, n_samples: int = N_SAMPLES) -> Dict:
    samples = []

    for _ in range(n_samples):
        try:
            pred = generate_single_cot_en(question)
            samples.append(pred)
        except Exception:
            continue

    if not samples:
        raise RuntimeError("All self-consistency samples failed")

    normalized_answers = [normalize_answer(sample.answer) for sample in samples]
    answer_counter = Counter(normalized_answers)

    final_answer_norm, final_count = answer_counter.most_common(1)[0]

    majority_samples = [
        sample for sample in samples
        if normalize_answer(sample.answer) == final_answer_norm
    ]

    representative_sample = majority_samples[0]

    return {
        "answer": representative_sample.answer,
        "reasoning_steps": representative_sample.reasoning_steps,
        "reason": representative_sample.reason,
        "all_sample_answers": [sample.answer for sample in samples],
        "all_sample_reasons": [sample.reason for sample in samples],
        "all_sample_reasoning_steps": [sample.reasoning_steps for sample in samples],
        "vote_distribution": dict(answer_counter),
        "selected_count": final_count,
        "num_valid_samples": len(samples),
    }


# ---------------------------
# 10. 전체 평가
# ---------------------------
results = []
correct = 0

for item in dataset:
    qid = item["id"]
    question = item["question"]
    expected = answer_map[qid]["expected_answer"]

    try:
        pred = solve_self_consistency_en(question, n_samples=N_SAMPLES)

        predicted_answer = normalize_answer(pred["answer"])
        expected_answer = normalize_answer(expected)
        is_correct = predicted_answer == expected_answer

        if is_correct:
            correct += 1

        results.append({
            "id": qid,
            "difficulty": item.get("difficulty"),
            "question": question,
            "predicted_answer": pred["answer"],
            "predicted_reasoning_steps": pred["reasoning_steps"],
            "predicted_reason": pred["reason"],
            "all_sample_answers": pred["all_sample_answers"],
            "all_sample_reasons": pred["all_sample_reasons"],
            "all_sample_reasoning_steps": pred["all_sample_reasoning_steps"],
            "vote_distribution": pred["vote_distribution"],
            "selected_count": pred["selected_count"],
            "num_valid_samples": pred["num_valid_samples"],
            "expected_answer": expected,
            "expected_reasoning": answer_map[qid].get("reasoning"),
            "is_correct": is_correct,
        })

    except Exception as e:
        results.append({
            "id": qid,
            "difficulty": item.get("difficulty"),
            "question": question,
            "predicted_answer": "ERROR",
            "predicted_reasoning_steps": [],
            "predicted_reason": str(e),
            "all_sample_answers": [],
            "all_sample_reasons": [],
            "all_sample_reasoning_steps": [],
            "vote_distribution": {},
            "selected_count": 0,
            "num_valid_samples": 0,
            "expected_answer": expected,
            "expected_reasoning": answer_map[qid].get("reasoning"),
            "is_correct": False,
        })

accuracy = correct / len(dataset)
print(f"Step 5 English Prompt Accuracy: {correct}/{len(dataset)} = {accuracy:.2%}")

Path("../outputs").mkdir(exist_ok=True)
with open("../outputs/step5_english_prompt_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)