import os
import json
from pathlib import Path
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 1. Pydantic 스키마
# ==========================================
class QuestionAnswer(BaseModel):
    q_id: str = Field(description="질문의 ID (예: q01)")
    answer: str = Field(description="본인부담률 또는 본인부담금 정답 (예: 5%, 1,000원, 30,000원)")

class EvaluationResult(BaseModel):
    results: list[QuestionAnswer] = Field(description="모든 질문에 대한 답변 리스트")

# ==========================================
# 2. 데이터 로드
# ==========================================
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 2025년 문항만 (source_year == "2025")
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
# 3. 2025년 섹션03 청크에서 본인부담금 표 데이터 구성
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
# 4. 프롬프트 구성 (Zero-shot)
# ==========================================
system_prompt = f"""아래는 2025년 의료급여 본인일부부담금 관련 데이터입니다.
이 데이터만을 바탕으로 질문에 대해 정확한 본인부담률 또는 본인부담금을 답하세요.
계산이 필요한 경우 계산 결과를 금액(원)으로 답하세요.
답만 간결하게 작성하세요.

=== 2025년 의료급여 본인일부부담금 참조 데이터 ===
{copayment_reference}
"""

user_prompt = "다음 질문들에 대해 정답을 작성하세요.\n\n"
for q in questions:
    user_prompt += f"ID: {q['id']} | 질문: {q['question']}\n"

# ==========================================
# 5. API 호출
# ==========================================
print("gpt-4o-mini Zero-shot 실행 중...\n")

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_format=EvaluationResult,
)

parsed = response.choices[0].message.parsed
model_answers = {item.q_id: item.answer for item in parsed.results}

# ==========================================
# 6. 채점 (의미 기반 — 핵심 값 포함 여부)
# ==========================================
def normalize(s: str) -> str:
    return s.replace(" ", "").replace(",", "").replace("원", "").replace("%", "").lower()

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

print("=" * 70)
print(f"{'ID':<5} | {'난이도':<10} | {'정답':<6} | {'모델 답변':<20} | 기대값")
print("-" * 70)

for q in questions:
    qid = q["id"]
    expected = q["expected"]
    model_ans = model_answers.get(qid, "답변 없음")
    ok = is_correct(expected, model_ans)
    if ok:
        correct_count += 1
    mark = "O" if ok else "X"
    print(f"{qid:<5} | {q['difficulty']:<10} | {mark:^6} | {model_ans:<20} | {expected}")

accuracy = correct_count / total * 100
print("=" * 70)
print(f"Zero-shot 정답률: {accuracy:.1f}% ({correct_count}/{total})")
print("=" * 70)
