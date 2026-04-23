import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# 1. 환경 설정
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini" 

# 2. 구조화된 출력(Structured Output)을 위한 Pydantic 모델
class CopaymentResponse(BaseModel):
    reason: str
    answer: str

# 3. LLM에게 제공할 지식 (의료급여 표 마크다운)
copayment_reference = """
[의료급여 본인부담률 참조 데이터]

1. 65세 이상 틀니 및 치과 임플란트 본인부담률
- 1종: 틀니 5%, 임플란트 10%
- 2종: 틀니 15%, 임플란트 20%
* 본인부담 보상제·상한제 해당되지 않음

2. 추나요법 본인부담률
- 디스크, 협착증: [1종] 복잡추나 30%, 단순/특수추나 30% | [2종] 복잡추나 40%, 단순/특수추나 40%
- 디스크, 협착증 외: [1종] 복잡추나 80%, 단순/특수추나 30% | [2종] 복잡추나 80%, 단순/특수추나 40%

3. 의료급여 2종수급권자 본인부담률 특례
(1) 15세 이하 아동
- 입원: 6세 미만(무료), 6세 이상~15세 이하(3%)
- 외래: 1세 미만(제1차 무료, 제2·3차 5%), 1세 미만 만성질환자(제2차 무료), 5세까지 조산아·저체중출생아(병원급 이상 5%)
(2) 분만 및 임신부
- 입원: 자연분만/제왕절개분만(무료), 고위험 임신부(5%)
- 외래: 임신부(병원급 이상 5%)
(3) 치아 홈메우기
- 입원: 16세 이상~18세 이하(5%), 6세 이상~15세 이하(3%), 6세 미만(무료)
- 외래: 18세 이하(병원급 이상 5%)
(4) 정신질환 외래진료
- 조현병(병원급 이상 5%), 조현병 외 정신질환(병원급 이상 10%)
(5) 치매질환 입원 및 병원급 이상 외래진료: 5%
(6) CT, MRI, PET 등
- 임신부(제1차 5%), 조산아 및 저체중출생아(제1차 5%), 치매(제1차 5%), 1세 미만 만성질환자(제2차 5%), 조현병 등 정신질환자(제2·3차 15%)
"""

# 4. 최적화: CoT + Few-shot (Temperature=0)
def run_optimized_cot_fewshot(question: str):
    system_prompt = f"""당신은 국민건강보험공단의 본인부담률 산정 전문가입니다. 
아래 의료급여 본인부담률 참조 데이터를 바탕으로 질문에 답하세요.

[추론 지침 - Chain of Thought]
최종 정답(answer)을 도출하기 전에 'reason' 필드에 반드시 다음 단계를 거쳐 생각하세요:
1단계: 수급자 종별(1종/2종) 파악
2단계: 환자의 연령 및 특이사항(임신, 질환 유무, 만성질환 등) 파악
3단계: 진료 항목(입원/외래, 치료 종류, 검사 종류) 및 의료기관 종별(1차/2차/3차) 파악
4단계: 참조 데이터에서 위 조건들에 완벽히 부합하는 본인부담률(%) 찾기 (일반 규칙보다 특수/예외 규칙 우선 적용)
5단계: 만약 진료비/치료비 금액이 주어졌다면, 찾은 비율을 곱하여 최종 본인부담금(원) 계산하기

[답변 작성 예시 (Few-shot)]
Q: 2종 수급권자인 생후 8개월 아기가 만성질환자이고 제2차의료급여기관에서 외래 진료를 받으면 본인부담률은?
A: {{"reason": "1단계: 2종. 2단계: 생후 8개월(1세 미만) 만성질환자. 3단계: 2차 외래. 4단계: 1세 미만 만성질환자 제2차 외래는 무료.", "answer": "무료"}}

Q: 1종 수급권자가 디스크로 복잡추나를 받았고 치료비가 100,000원입니다. 본인부담금은?
A: {{"reason": "1단계: 1종. 2단계: 디스크. 3단계: 복잡추나. 4단계: 디스크 1종 복잡추나는 30%. 5단계: 100,000 * 0.3 = 30,000원.", "answer": "30,000원"}}

Q: 65세 이상 2종 수급권자가 틀니(시술비 1,000,000원)와 임플란트(시술비 1,000,000원)를 동시에 하면?
A: {{"reason": "1단계: 2종. 2단계: 65세 이상. 3단계: 틀니, 임플란트. 4단계: 2종 틀니 15%, 임플란트 20%. 5단계: 틀니 150,000원, 임플란트 200,000원.", "answer": "틀니 150,000원, 임플란트 200,000원"}}

Q: 65세 이상 2종 수급권자가 임플란트를 할 때 본인부담 보상제나 상한제가 적용되나요?
A: {{"reason": "1단계: 2종. 2단계: 65세 이상. 3단계: 임플란트. 4단계: 틀니/임플란트는 본인부담 보상제 및 상한제 제외 항목임.", "answer": "해당되지 않음"}}

{copayment_reference}
"""
    try:
        response = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Q: {question}"}
            ],
            response_format=CopaymentResponse,
            temperature=0, # 창의성을 완전히 배제하고 논리에만 집중
        )
        return response.choices[0].message.parsed
    except Exception as e:
        print(f"Error: {e}")
        return None

# 5. 실행 및 평가 로직
def run_experiment():
    with open('data/dataset.jsonl', 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
    with open('data/answer_key.jsonl', 'r', encoding='utf-8') as f:
        answers = {json.loads(line)['id']: json.loads(line) for line in f}

    print("--- [CoT + Few-shot (Temp=0)] 실행 중... ---\n")
    
    correct_count = 0
    total_count = len(questions)

    for q in questions:
        q_id = q['id']
        question_text = q['question']
        difficulty = q['difficulty']
        expected_ans = answers[q_id]['expected_answer']

        # LLM 호출
        result = run_optimized_cot_fewshot(question_text)
        if not result: continue

        actual_ans = result.answer
        
        # 정답 비교
        is_correct = expected_ans.replace(" ", "") in actual_ans.replace(" ", "")
        
        if is_correct:
            correct_count += 1
            print(f"[{q_id}][{difficulty}] ✅ 정답: {expected_ans}")
        else:
            print(f"[{q_id}][{difficulty}] ❌ 오답")
            print(f"  - LLM 추론: {result.reason}")
            print(f"  - LLM 답변: {actual_ans} (실제 정답: {expected_ans})")
    
    # 최종 결과 출력
    accuracy = (correct_count / total_count) * 100
    print("\n" + "="*40)
    print(f"최종 최적화 정답률: {correct_count}/{total_count} ({accuracy:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    run_experiment()