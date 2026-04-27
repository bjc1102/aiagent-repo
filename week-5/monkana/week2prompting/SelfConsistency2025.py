import os
import json
from pathlib import Path
from collections import Counter
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

NUM_SAMPLES = 5   # 동일 질문을 몇 번 샘플링할지

# ==========================================
# 1. Pydantic 스키마
# ==========================================
class QuestionAnswer(BaseModel):
    q_id: str = Field(description="질문의 ID (예: q01)")
    reasoning: str = Field(description="단계별 추론 과정")
    answer: str = Field(description="최종 본인부담률 또는 본인부담금 (예: 5%, 1,000원, 30,000원)")

class EvaluationResult(BaseModel):
    results: list[QuestionAnswer] = Field(description="모든 질문에 대한 답변 리스트")

# ==========================================
# 2. 데이터 로드
# ==========================================
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

questions = []
with open(BASE_DIR / "goldenDataset.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        if data.get("source_year") == "2025":
            questions.append({
                "id": data["id"],
                "question": data["question"],
                "expected": data["expected_answer"],
                "difficulty": data["difficulty"],
            })

# ==========================================
# 3. 청크 데이터 로드
# ==========================================
with open(BASE_DIR / "all_chunks(2025).json", encoding="utf-8") as f:
    chunks = json.load(f)

sec03 = [c for c in chunks if "03 의료급여 본인일부부담금" in c.get("parent_subject", "")]

def format_chunk(c: dict) -> str:
    lines = [f"[{c.get('subject', '')}]"]
    text = c.get("text", "").strip()
    if text:
        lines.append(text)
    for t in c.get("tables", []):
        lines.append(str(t))
    return "\n".join(lines)

copayment_reference = "\n\n".join(format_chunk(c) for c in sec03)

# ==========================================
# 4. 프롬프트 (CoT 기반 — 다양한 reasoning path 유도)
# ==========================================
system_prompt = f"""아래는 2025년 의료급여 본인일부부담금 관련 데이터입니다.
각 질문에 대해 수급권자 종류, 진료 유형, 기관 차수, 특수 조건을 단계별로 확인한 뒤
표에서 해당 본인부담률을 찾고 필요시 계산하여 최종 답을 도출하세요.
reasoning 필드에 추론 과정을, answer 필드에 최종 답만 간결하게 작성하세요.

=== 2025년 의료급여 본인일부부담금 참조 데이터 ===
{copayment_reference}
"""

user_prompt = "다음 질문들에 대해 단계별로 추론 후 정답을 작성하세요.\n\n"
for q in questions:
    user_prompt += f"ID: {q['id']} | 질문: {q['question']}\n"

# ==========================================
# 5. NUM_SAMPLES 번 샘플링 (temperature > 0 으로 다양성 확보)
# ==========================================
print(f"gpt-4o-mini Self-Consistency 실행 중... ({NUM_SAMPLES}회 샘플링)\n")

all_samples: dict[str, list[str]] = {q["id"]: [] for q in questions}

for i in range(NUM_SAMPLES):
    print(f"  샘플 {i+1}/{NUM_SAMPLES}...")
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=EvaluationResult,
        temperature=0.7,
    )
    parsed = response.choices[0].message.parsed
    for item in parsed.results:
        if item.q_id in all_samples:
            all_samples[item.q_id].append(item.answer.strip())

# ==========================================
# 6. 다수결 집계
# ==========================================
def normalize(s: str) -> str:
    return s.replace(" ", "").replace(",", "").replace("원", "").replace("%", "").lower()

def majority_vote(answers: list[str]) -> tuple[str, Counter]:
    counts: Counter = Counter(normalize(a) for a in answers)
    top_norm = counts.most_common(1)[0][0]
    # 원본 형태 복원 (첫 번째 매칭)
    for a in answers:
        if normalize(a) == top_norm:
            return a, counts
    return answers[0], counts

def is_correct(expected: str, model_ans: str) -> bool:
    e = normalize(expected)
    m = normalize(model_ans)
    if e == m:
        return True
    if e in m or m in e:
        return True
    return False

correct_count = 0
total = len(questions)

print()
print("=" * 80)
print(f"{'ID':<5} | {'난이도':<10} | {'정답':<6} | {'최종답(다수결)':<22} | {'분포':<20} | 기대값")
print("-" * 80)

for q in questions:
    qid = q["id"]
    expected = q["expected"]
    answers = all_samples.get(qid, [])
    if not answers:
        final_ans, counts = "답변 없음", Counter()
    else:
        final_ans, counts = majority_vote(answers)

    ok = is_correct(expected, final_ans)
    if ok:
        correct_count += 1
    mark = "O" if ok else "X"
    dist = " / ".join(f"{k}:{v}" for k, v in counts.most_common(3))
    print(f"{qid:<5} | {q['difficulty']:<10} | {mark:^6} | {final_ans:<22} | {dist:<20} | {expected}")

accuracy = correct_count / total * 100
print("=" * 80)
print(f"Self-Consistency 정답률: {accuracy:.1f}% ({correct_count}/{total})  [샘플 {NUM_SAMPLES}회]")
print("=" * 80)
