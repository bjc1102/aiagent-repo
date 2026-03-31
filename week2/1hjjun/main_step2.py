import os
import json
from typing import List
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class LLMResponse(BaseModel):
    answer: str
    reasoning: str

# 1. Few-shot 예시 정의
FEW_SHOT_EXAMPLES = """
[질문-답변 가이드 예시]

예시 1 (특례 적용으로 인한 무료 케이스)
질문: 1종 수급권자인 5세 미만 아동이 상급종합병원에 입원하여 진료를 받으면 본인부담률은 얼마인가요?
답변: {
    "answer": "무료",
    "reasoning": "6세 미만 아동은 모든 의료급여기관에서 입원 시 본인부담금이 면제되는 '무료' 대상입니다."
}

예시 2 (보상제/상한제 제외 케이스)
질문: 1종 수급권자가 추나요법 시술을 받을 때 본인부담 상한제가 적용되나요?
답변: {
    "answer": "해당되지 않음",
    "reasoning": "참조 데이터에 따르면 추나요법은 본인부담 보상제 및 상한제 적용 제외 항목으로 명시되어 있습니다."
}

예시 3 (간결한 수치 출력 케이스)
질문: 65세 이상 2종 수급권자가 틀니(시술비 1,200,000원)와 임플란트(시술비 900,000원)를 동시에 할 경우 각각의 본인부담금은 얼마인가요?
답변: { "answer": "틀니 180,000원, 임플란트 180,000원", "reasoning": "2종 수급권자의 틀니 본인부담률은 15%(1,200,000 * 0.15 = 180,000원), 임플란트는 20%(900,000 * 0.20 = 180,000원)입니다." }
"""

COPAYMENT_REFERENCE = """
# [의료급여 본인부담률 핵심 참조 가이드]

## 1. 수급권자 및 기관 정의
- 1종 수급권자: 근로무능력가구, 희귀/중증난치질환자, 시설수급자 등
- 2종 수급권자: 기초생활수급자 중 1종 기준에 해당하지 않는 자
- 의료기관 종별: 1차(의원), 2차(병원/종합병원), 3차(상급종합병원)[cite: 1]

## 2. 기본 본인부담률 (일반 진료)
| 구분 | 1차(의원) | 2차(병원) | 3차(상급) | 약국 | CT/MRI/PET |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1종 입원** | 무료 | 무료 | 무료 | - | 무료 |
| **1종 외래** | 1,000원 | 1,500원 | 2,000원 | 500원 | 5% |
| **2종 입원** | 10% | 10% | 10% | - | 10% |
| **2종 외래** | 1,000원 | 15% | 15% | 500원 | 15% |
* 선택의료급여기관 미신청자: 입원 20%, 외래/약국 30% 적용[cite: 1]

## 3. 연령별 특례 (최우선 적용)
- **1세 미만 (0세):** 입원/외래 모두 무료. 단, 2·3차 기관 외래진료 이거나 특수검사(CT/MRI/PET)는 5%[cite: 1]
- **1세 미만 만성질환자:** 2차 기관 외래 진료 시 본인부담 무료[cite: 1]
- **6세 미만:** 입원 무료[cite: 1]
- **6세 이상 ~ 15세 이하:** 입원 3%[cite: 1]
- **18세 이하:** 치아 홈메우기 입원(6세 미만 무료, 6~15세 3%, 16~18세 5%), 외래(병원급 이상 5%)[cite: 1]

## 4. 질환 및 항목별 특례
- **노인 (65세 이상):** 
    - 틀니: 1종 5%, 2종 15%
    - 임플란트: 1종 10%, 2종 20%
  * ※ 본인부담 보상제/상한제 적용 제외 항목임[cite: 1]
- **정신질환 (외래):** 
    - 조현병: 병원급 이상 5%
    - 조현병 외 정신질환: 병원급 이상 10%[cite: 1]
- **치매질환:** 입원 및 병원급 이상 외래 5%[cite: 1]
- **분만 및 임신부:** 
    - 입원: 자연분만/제왕절개 무료, 고위험 임신부 5%
    - 외래: 임신부(유산/사산 포함) 병원급 이상 5%[cite: 1]
- **추나요법:** 
    - 디스크/협착증: 1종 30%, 2종 40% (단순/복잡 공통)
    - 디스크/협착증 외: 1종/2종 모두 80% (복잡추나 기준)[cite: 1]

## 5. 특수검사 (CT, MRI, PET) 상세
- 임신부, 5세 이하 조산아/저체중아, 치매질환자: 1차 기관 5%
- 1세 미만 만성질환자: 2차 기관 5%
- 조현병 등 정신질환자: 2·3차 기관 15%[cite: 1]

## 6. 기타 원칙
- 본인부담 면제자: 18세 미만, 임산부, 희귀질환자, 1세 미만(의원급) 등[cite: 1]
- 식대 본인부담: 2종 장애인 20%, 중증질환자 5%, 6세 미만/자연분만 무료[cite: 1]
"""

def get_gemini_response(question: str):
    # COPAYMENT_REFERENCE는 이전과 동일하게 유지
    system_prompt = f"""
    당신은 의료급여 본인부담률 산정 전문가입니다.
    아래 참조 데이터를 바탕으로 질문에 정확한 본인부담률을 답하세요.
    최종 답변(answer)은 정답지에 명시된 대로 불필요한 수식어 없이 간결하게 작성하세요.

    {FEW_SHOT_EXAMPLES}

    [참조 데이터]
    {COPAYMENT_REFERENCE}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
        config={
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_schema": LLMResponse,
            "temperature": 0,
        }
    )
    return json.loads(response.text)

def run():
    results = []
    correct_count = 0
    
    # 1. 정답 로드
    answers = {}
    with open("data/answer_key.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            answers[item["id"]] = item["expected_answer"]

    # 2. 30문항 전체 로드
    with open("data/dataset.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    total_count = len(lines)
    print(f"\n🚀 Step 2: Few-shot (가상 예시 적용) - 총 {total_count}문항 전체 평가")
    print("-" * 50)

    for line in lines:
        data = json.loads(line)
        qid = data["id"]
        question = data["question"]
        expected = answers.get(qid)
        
        try:
            # 새로 만든 가상 예시가 포함된 함수 호출
            prediction = get_gemini_response(question) 
            predicted_ans = prediction["answer"]
            
            # 정답 비교 (공백 제거)
            is_correct = str(predicted_ans).replace(" ", "") == str(expected).replace(" ", "")
            
            if is_correct:
                correct_count += 1
                status = "✅"
            else:
                status = "❌"

            print(f"[{qid}] {status} | 예측: {predicted_ans} | 정답: {expected}")
            
            results.append({
                "id": qid,
                "is_correct": is_correct,
                "predicted": predicted_ans,
                "expected": expected
            })

        except Exception as e:
            print(f"[{qid}] 에러: {e}")

    # 최종 결과 리포트
    accuracy = (correct_count / total_count) * 100
    print(f"\n📊 [Step 2 결과 리포트]")
    print(f"전체 문항: {total_count} / 정답 수: {correct_count}")
    print(f"최종 정답률: {accuracy:.2f}%")

    # 4. 파일 저장
    with open("results/results_step2.json", "w", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=4)
    print("💾 결과가 'results/results_step2.json'에 저장되었습니다.")
if __name__ == "__main__":
    run()