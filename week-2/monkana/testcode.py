import os
import re
import json
from pydantic import BaseModel, Field
from openai import OpenAI

# ==========================================
# 1. Pydantic 모델 정의
# ==========================================
class QuestionAnswer(BaseModel):
    q_id: str = Field(description="질문의 ID (예: q01, q02)")
    step1_patient_info: str = Field(description="[1단계] 환자 정보 파악: 수급권자 종류(1종/2종), 나이/상태, 진료 유형(입원/외래), 이용 기관을 정리")
    step2_rule_lookup: str = Field(description="[2단계] 적용 규정 탐색: 참조 데이터에서 해당 섹션과 조건을 찾아 본인부담률 명시. CT/MRI/PET는 반드시 CT_MRI_PET 섹션 우선 확인")
    step3_calculation: str = Field(description="[3단계] 계산: 금액이 있으면 진료비 × 본인부담률 수식으로 계산. 항목이 여러 개면 각각 계산")
    answer: str = Field(description="[최종 답변] 본인부담률 또는 본인부담금만 간결하게 (예: 5%, 무료, 60,000원)")

class EvaluationResult(BaseModel):
    results: list[QuestionAnswer] = Field(description="모든 질문에 대한 답변 리스트")

# ==========================================
# 2. 설정 및 초기화
# ==========================================
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DATA_DIR = r"C:/Users/sjn01/Desktop/새 폴더 (4)/week-2/monkana/data"
DATASET_PATH = os.path.join(DATA_DIR, "dataset.jsonl")
ANSWER_KEY_PATH = os.path.join(DATA_DIR, "answer_key.jsonl")

questions = []
answers_key = {}

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        questions.append({"id": data["id"], "question": data["question"]})

with open(ANSWER_KEY_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        answers_key[data["id"]] = data["expected_answer"]

# ==========================================
# 3. 프롬프트 구성
# ==========================================
copayment_reference = json.dumps(
[{'카테고리': '치과', '수급종': '1종', '항목': '틀니', '본인부담률': '5%'}, {'카테고리': '치과', '수급종': '1종', '항목': '임플란트', '본인부담률': '10%'}, {'카테고리': '치과', '수급종': '2종', '항목': '틀니', '본인부담률': '15%'}, {'카테고리': '치과', '수급종': '2종', '항목': '임플란트', '본인부담률': '20%'}, {'카테고리': '치과', '비고': '본인부담 보상제·상한제 해당 없음'}, {'카테고리': '추나', '수급종': '1종', '질환': '디스크·협착증', '추나종류': '복잡추나', '본인부담률': '30%'}, {'카테고리': '추나', '수급종': '2종', '질환': '디스크·협착증', '추나종류': '복잡추나', '본인부담률': '40%'}, {'카테고리': '추나', '수급종': '1종', '질환': '디스크·협착증', '추나종류': '단순·특수추나', '본인부담률': '30%'}, {'카테고리': '추나', '수급종': '2종', '질환': '디스크·협착증', '추나종류': '단순·특수추나', '본인부담률': '40%'}, {'카테고리': '추나', '수급종': '1종', '질환': '디스크·협착증 외', '추나종류': '복잡추나', '본인부담률': '80%'}, {'카테고리': '추나', '수급종': '2종', '질환': '디스크·협착증 외', '추나종류': '복잡추나', '본인부담률': '80%'}, {'카테고리': '추나', '수급종': '1종', '질환': '디스크·협착증 외', '추나종류': '단순·특수추나', '본인부담률': '30%'}, {'카테고리': '추나', '수급종': '2종', '질환': '디스크·협착증 외', '추나종류': '단순·특수추나', '본인부담률': '40%'}, {'카테고리': '치아홈메우기', '진료구분': '입원', '조건': '16세 이상~18세 이하', '본인부담률': '5%'}, {'카테고리': '치아홈메우기', '진료구분': '입원', '조건': '6세 이상~15세 이하', '본인부담률': '3%'}, {'카테고리': '치아홈메우기', '진료구분': '입원', '조건': '6세 미만', '본인부담률': '무료'}, {'카테고리': '치아홈메우기', '진료구분': '외래', '조건': '18세 이하', '기관': '병원급 이상', '본인부담률': '5%'}, {'카테고리': '분만·임신부', '진료구분': '입원', '조건': '자연분만', '본인부담률': '무료'}, {'카테고리': '분만·임신부', '진료구분': '입원', '조건': '제왕절개분만', '본인부담률': '무료'}, {'카테고리': '분만·임신부', '진료구분': '입원', '조건': '고위험 임신부', '본인부담률': '5%'}, {'카테고리': '분만·임신부', '진료구분': '외래', '조건': '임신부(유산·사산 포함)', '기관': '병원급 이상', '본인부담률': '5%'}, {'카테고리': '아동', '진료구분': '입원', '조건': '6세 미만', '본인부담률': '무료'}, {'카테고리': '아동', '진료구분': '입원', '조건': '6세 이상~15세 이하', '본인부담률': '3%'}, {'카테고리': '아동', '진료구분': '외래', '조건': '1세 미만', '기관': '제1차의료급여기관', '본인부담률': '무료'}, {'카테고리': '아동', '진료구분': '외래', '조건': '1세 미만', '기관': '제2·3차의료급여기관', '본인부담률': '5%'}, {'카테고리': '아동', '진료구분': '외래', '조건': '1세 미만 만성질환자', '기관': '제2차의료급여기관', '본인부담률': '무료'}, {'카테고리': '아동', '진료구분': '외래', '조건': '5세까지 조산아·저체중출생아', '기관': '병원급 이상', '본인부담률': '5%'}, {'카테고리': '정신질환', '진료구분': '외래', '조건': '조현병', '기관': '병원급 이상', '본인부담률': '5%'}, {'카테고리': '정신질환', '진료구분': '외래', '조건': '조현병 외 정신질환', '기관': '병원급 이상', '본인부담률': '10%'}, {'카테고리': '정신질환', '진료구분': '외래', '조건': '장기지속형 주사제 (1·2종 공통, 외래본인부담면제자 제외)', '본인부담률': '5%'}, {'카테고리': '치매', '진료구분': '입원', '본인부담률': '5%'}, {'카테고리': '치매', '진료구분': '외래', '기관': '병원급 이상', '본인부담률': '5%'}, {'카테고리': 'CT·MRI·PET', '조건': '임신부(유산·사산 포함)', '기관': '제1차의료급여기관', '본인부담률': '5%'}, {'카테고리': 'CT·MRI·PET', '조건': '5세까지 조산아·저체중출생아', '기관': '제1차의료급여기관', '본인부담률': '5%'}, {'카테고리': 'CT·MRI·PET', '조건': '치매 질환자', '기관': '제1차의료급여기관', '본인부담률': '5%'}, {'카테고리': 'CT·MRI·PET', '조건': '1세 미만 만성질환자', '기관': '제2차의료급여기관', '본인부담률': '5%'}, {'카테고리': 'CT·MRI·PET', '조건': '조현병 등 정신질환자', '기관': '제2·3차의료급여기관', '본인부담률': '15%'}],
    ensure_ascii=False,
    indent=2
)

system_prompt = f"""당신은 의료급여 본인부담률 전문가입니다.
아래 참조 데이터는 규칙 하나당 객체 하나인 플랫 리스트입니다.
각 질문에 대해 반드시 다음 3단계 순서로 사고한 뒤 최종 답변을 작성하세요.

[사고 순서]
1. step1_patient_info : 환자 정보 파악
   → 수급권자 종류(1종/2종), 나이·상태, 입원/외래 구분, 이용 기관을 정리

2. step2_rule_lookup  : 규정 탐색
   → 참조 데이터에서 카테고리·조건·기관이 일치하는 항목을 찾아 본인부담률 명시
   → ※ CT·MRI·PET 검사는 반드시 카테고리 "CT·MRI·PET" 항목만 적용 (일반 외래 규정 사용 금지)

3. step3_calculation  : 계산
   → 본인부담률 확정 후, 금액이 주어진 경우 수식으로 계산 (진료비 × 본인부담률 = 본인부담금)
   → 항목이 여러 개면 각각 따로 계산

4. answer : 최종 답변만 간결하게 (예: 5%, 무료, 60,000원)

[참조 데이터]
{copayment_reference}
"""

user_prompt = "다음 질문들에 대해 3단계로 사고한 뒤 각각 답하세요.\n\n"
for q in questions:
    user_prompt += f"ID: {q['id']} | 질문: {q['question']}\n"

# ==========================================
# 4. API 호출
# ==========================================
print("🚀 GPT-4o-mini에게 요청 중입니다...\n")

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_format=EvaluationResult,
    temperature=0
)

parsed_output = response.choices[0].message.parsed
model_answers = {item.q_id: item for item in parsed_output.results}

# ==========================================
# 5. 채점 로직
# ==========================================

# 동의어 그룹: 같은 의미로 취급할 표현들
SYNONYM_GROUPS = [
    {'해당되지않음', '미적용', '아니오', '없음', '적용안됨', '해당없음', '적용되지않음'},
]

def normalize_synonyms(text: str) -> str:
    """동의어를 그룹의 첫 번째 단어로 통일"""
    clean = text.replace(" ", "")
    for group in SYNONYM_GROUPS:
        canonical = sorted(group)[0]  # 그룹 내 정렬 첫 번째를 대표값으로
        for word in group:
            if word in clean:
                clean = clean.replace(word, canonical)
    return clean


def extract_values(text: str) -> set:
    """
    텍스트에서 핵심 값(%, 원, 무료 등)을 집합으로 추출.
    ※ 반드시 숫자로 시작하는 패턴만 잡아서 ',36,000원' 같은 오매칭 방지
    """
    clean = text.replace(" ", "")

    # 퍼센트: 숫자로 시작, 반드시 % 로 끝
    percentages = set(re.findall(r'\d[\d,]*%', clean))

    # 금액: 숫자로 시작 (쉼표 포함), 원으로 끝  ← 핵심 수정: \d 로 시작
    currency = set(re.findall(r'\d[\d,]*원', clean))

    # 키워드
    keywords = set()
    for group in SYNONYM_GROUPS:
        canonical = sorted(group)[0]
        if any(word in clean for word in group):
            keywords.add(canonical)
    if '무료' in clean:
        keywords.add('무료')

    return percentages | currency | keywords


def judge(model_ans: str, expected: str) -> tuple[bool, str]:
    """채점 함수 — (정답 여부, 판단 근거) 반환"""

    # 1단계: 구분자 정규화 후 완전 일치
    def normalize(s):
        return re.sub(r'[\s/,·]+', ',', s.replace(" ", "")).strip(',')

    if normalize(model_ans) == normalize(expected):
        return True, "완전일치"

    # 2단계: 동의어 정규화 후 완전 일치
    if normalize_synonyms(model_ans) == normalize_synonyms(expected):
        return True, "동의어일치"

    # 3단계: 핵심 값 집합 비교 (포맷 차이 허용)
    model_vals    = extract_values(model_ans)
    expected_vals = extract_values(expected)

    if model_vals and expected_vals:
        if model_vals == expected_vals:
            return True, "값집합일치"
        # 동의어 정규화 후 집합 비교
        model_syn    = extract_values(normalize_synonyms(model_ans))
        expected_syn = extract_values(normalize_synonyms(expected))
        if model_syn == expected_syn:
            return True, "동의어+값"

    return False, "불일치"


# ==========================================
# 6. 결과 출력
# ==========================================
correct_count   = 0
total_questions = len(questions)
wrong_list      = []

print("=" * 95)
print(f"{'ID':<5} | {'결과':^4} | {'판단':^8} | {'모델 답변':<22} | {'정답':<22} | CoT 추론")
print("-" * 95)

for q in questions:
    q_id      = q["id"]
    expected  = answers_key.get(q_id, "")
    item      = model_answers.get(q_id)
    model_ans = item.answer if item else "답변 없음"
    cot_summary = f"[1]{item.step1_patient_info} / [2]{item.step2_rule_lookup} / [3]{item.step3_calculation}" if item else ""

    correct, judge_reason = judge(model_ans, expected)

    if correct:
        correct_count += 1
        mark = "O"
    else:
        mark = "X"
        wrong_list.append((q_id, model_ans, expected, item))

    print(f"{q_id:<5} | {mark:^4} | {judge_reason:<8} | {model_ans:<22} | {expected:<22} | {cot_summary[:80]}")

# ==========================================
# 7. 최종 정답률 및 오답 요약
# ==========================================
accuracy = (correct_count / total_questions) * 100
print("=" * 95)
print(f"🎯 최종 정답률: {accuracy:.2f}% ({correct_count}/{total_questions})")
print("=" * 95)

if wrong_list:
    print("\n📋 오답 목록 (CoT 추론 포함):")
    for q_id, model_ans, expected, item in wrong_list:
        print(f"\n  [{q_id}]")
        print(f"    모델 answer  : {model_ans}")
        print(f"    정답         : {expected}")
        if item:
            print(f"    [1] 환자정보 : {item.step1_patient_info}")
            print(f"    [2] 규정탐색 : {item.step2_rule_lookup}")
            print(f"    [3] 계산     : {item.step3_calculation}")

usage = response.usage
print(f"\n토큰 사용량 → 전체: {usage.total_tokens} | "
      f"추론: {usage.completion_tokens_details.reasoning_tokens}")