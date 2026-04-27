import os
import json
from pydantic import BaseModel, Field
from openai import OpenAI

# ==========================================
# 1. Pydantic 모델 정의 (출력 스키마 강제)
# ==========================================
class QuestionAnswer(BaseModel):
    q_id: str = Field(description="질문의 ID (예: q01, q02)")
    answer: str = Field(description="본인부담률 또는 본인부담금 정답 (예: 5%, 무료, 60,000원)")

class EvaluationResult(BaseModel):
    results: list[QuestionAnswer] = Field(description="모든 질문에 대한 답변 리스트")

# ==========================================
# 2. 설정 및 초기화
# ==========================================
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 파일 경로 설정
DATA_DIR = r"D:\aiagent(personal)\aiagent-repo\week2\data"
DATASET_PATH = os.path.join(DATA_DIR, "dataset.jsonl")
ANSWER_KEY_PATH = os.path.join(DATA_DIR, "answer_key.jsonl")

questions = []
answers_key = {}

# 데이터 로드
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        questions.append({"id": data["id"], "question": data["question"]})

with open(ANSWER_KEY_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        answers_key[data["id"]] = data["expected_answer"]

# ==========================================
# 3. 프롬프트 구성 (출력 제어 문구 삭제 -> 순수 Zero-shot)
# ==========================================
copayment_reference = """
01. 65세 이상 틀니 및 치과 임플란트 본인부담률 1종: 틀니 5% / 임플란트 10% 2종: 틀니 15% / 임플란트 20% 참고: 본인부담 보상제·상한제 해당되지 않음, 2종 장애인의 경우 장애인의료비 지원 없음 
02. 추나요법 본인부담률 디스크, 협착증:복잡추나: 1종 30% / 2종 40% 단순추나, 특수추나: 1종 30% / 2종 40% 디스크, 협착증 외:복잡추나: 1종 80% / 2종 80% 단순추나, 특수추나: 1종 30% / 2종 40% 
03. 의료급여 2종수급권자 본인부담률 치아 홈메우기 (입원): 16세 이상~18세 이하 5% / 6세 이상~15세 이하 3% / 6세 미만 무료 치아 홈메우기 (외래): 18세 이하 병원급 이상 5% 분만 및 임신부 (입원): 자연분만 무료 / 제왕절개분만 무료 / 고위험 임신부 5% 분만 및 임신부 (외래): 임신부(유산·사산 포함) 병원급 이상 5% 

04. 15세 이하 아동 
입원: 6세 미만 무료 / 6세 이상~15세 이하 3% 
외래:1세 미만: 제1차의료급여기관 무료 / 제2·3차의료급여기관 5% 1세 미만 만성질환자: 제2차의료급여기관 무료 5세까지의 조산아·저체중출생아: 병원급 이상 5% 
정신질환 외래진료 조현병: 병원급 이상 5% / 조현병 외 정신질환: 병원급 이상 10% / 장기지속형 주사제: 5% (1·2종 모두 해당, 외래본인부담면제자 제외) 

05. 기타 질환
치매질환: 입원 및 병원급 이상 외래진료 5% 

06. CT, MRI, PET 등 
임신부(유산·사산 포함): 제1차의료급여기관 5%
5세까지의 조산아 및 저체중 출생아: 제1차의료급여기관 5%
치매 질환자: 제1차의료급여기관 5%
1세 미만 만성질환자: 제2차의료급여기관 5%
조현병 등 정신질환자: 제2·3차의료급여기관 15%
"""

system_prompt = f"""아래는 의료급여 본인부담률 참조 데이터입니다.
질문에 대해 정확한 본인부담률을 답하세요. 답만 간결하게 작성하세요.

{copayment_reference}
"""

user_prompt = "다음 질문들에 대해 정답을 작성하세요.\n\n"
for q in questions:
    user_prompt += f"ID: {q['id']} | 질문: {q['question']}\n"

# ==========================================
# 4. API 호출 (Pydantic 스키마 강제)
# ==========================================
print("🚀 GPT-5o-mini에게 Pydantic 스키마로 30문제를 요청 중입니다...\n")

# 중요: 일반 completions.create가 아닌 beta.chat.completions.parse 사용
response = client.beta.chat.completions.parse(
    model="gpt-5-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_format=EvaluationResult, # 여기서 Pydantic 모델을 주입!
    #temperature=0,
)

# 파싱된 객체를 바로 가져옴 (문자열 split 같은 파싱 노가다 불필요)
parsed_output = response.choices[0].message.parsed

# ==========================================
# 5. 결과 채점
# ==========================================
correct_count = 0
total_questions = len(questions)

print("="*60)
print(f"{'ID':<5} | {'정답 여부':<6} | {'모델 답변':<20} | {'실제 정답'}")
print("-" * 60)

# 파싱된 결과를 딕셔너리 형태로 변환하여 쉽게 매칭
model_answers = {item.q_id: item.answer for item in parsed_output.results}

for q in questions:
    q_id = q["id"]
    expected = answers_key.get(q_id, "")
    model_ans = model_answers.get(q_id, "답변 없음")
    
    # 1. 띄어쓰기 제거 및 특수기호(/를 ,로) 통일
    norm_expected = expected.replace(" ", "").replace("/", ",")
    norm_model = model_ans.replace(" ", "").replace("/", ",")
    
    # 2. 유연한 정답 판별 로직
    is_correct = "X"
    if norm_expected == norm_model:
        is_correct = "O"
    elif norm_model in norm_expected or norm_expected in norm_model:
        # "10%"가 "병원급이상10%" 안에 포함되어 있으면 정답 인정
        is_correct = "O"
    elif ("미적용" in norm_model and "해당되지않음" in norm_expected):
        # 의미가 같은 특정 단어 예외 처리
        is_correct = "O"
        
    if is_correct == "O":
        correct_count += 1
        
    print(f"{q_id:<5} | {is_correct:^9} | {model_ans:<20} | {expected}")

# ==========================================
# 6. 최종 정답률 출력
# ==========================================
accuracy = (correct_count / total_questions) * 100
print("="*60)
print(f"🎯 최종 정답률: {accuracy:.2f}% ({correct_count}/{total_questions})")
print("="*60)