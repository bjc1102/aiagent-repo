import os
import json
import time
from collections import Counter
from typing import List
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv

# 1. 환경 설정 및 클라이언트 초기화
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 2. 결과 저장 폴더 자동 생성
os.makedirs("results", exist_ok=True)

# 3. Pydantic 모델 (최대한 간결하게 유지하여 에러율 감소)
class CoTResponse(BaseModel):
    step_1_category: str = Field(description="수급권자 종별 확인 (1종 또는 2종)")
    step_2_age_check: str = Field(description="연령 특례 해당 여부 확인")
    step_3_is_chronic: str = Field(description="만성질환자라는 명시적 조건이 있는지 확인")
    step_4_condition_check: str = Field(description="질환(치매, 조현병 등) 또는 항목(틀니, 추나 등) 특례 확인")
    step_5_institution_check: str = Field(description="의료기관 종별(1, 2, 3차) 확인")
    reasoning: str = Field(description="위 단계들을 종합한 최종 판단 근거")
    answer: str = Field(description="최종 정답 (예: 5%, 무료, 10,000원, 병원급 이상 10% 등)")

# 3. 의료급여 본인부담률 참조 데이터 (PDF 내용을 텍스트로 정리한 예시)
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

# 5. Self-Consistency 추론 함수
def get_self_consistency_response(question: str, n=5):
    system_prompt = f"""
    당신은 의료급여 본인부담률 산정 전문가입니다. 
    질문에 답하기 전, 반드시 다음 4단계 과정을 거쳐 사고하세요:

    1단계: 수급권자가 1종인지 2종인지 파악한다.
    2단계: 연령 특례(1세 미만, 6세 미만, 15세 이하 등)가 있는지 확인한다.
    3단계: 만성질환자라는 명시적 조건이 있는지 확인한다.
    4단계: 특정 질환(조현병, 치매 등)이나 특정 시술(틀니, 임플란트, 추나 등)에 대한 예외 규정이 있는지 확인한다.
    5단계: 방문한 의료기관(1차, 2차, 3차)에 따른 차등이 있는지 확인한다.
    답변이 두 개일 시 틀니 1,000원, 임플란트 2,000원 과 같이 간결하게 작성한다.
    적용되지 않을때는 "해당되지 않음" 으로 답변한다.
    부담금에 대한 질문은 원으로 답변하고 비율 질문은 %로 답변한다. 답변은 표 안에 있는 문구 그 자체로 작성한다. 예를 들어 "무료", "1,000원", "5%", "병원급 이상 10%" 등으로 답변한다.

    최종 답변(answer)은 이전과 같이 불필요한 수식어 없이 '결과값'만 적으세요.
    
    [참조 데이터]
    {COPAYMENT_REFERENCE}
    """

    candidate_answers = []
    
    for i in range(n):
        try:
            # API 호출 간 짧은 휴식 (Rate Limit 방지)
            if i > 0: time.sleep(1) 
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=question,
                config={
                    "system_instruction": system_prompt,
                    "response_mime_type": "application/json",
                    "response_schema": CoTResponse,
                    "temperature": 0.7, 
                }
            )
            
            if response.text:
                res_data = json.loads(response.text)
                clean_ans = str(res_data.get("answer", "")).replace(" ", "").strip()
                if clean_ans:
                    candidate_answers.append(clean_ans)
        
        except Exception as e:
            print(f" (시도 {i+1} 에러: {e})", end="")
            continue

    if not candidate_answers:
        return {"answer": "Error", "candidates": [], "reasoning": "모든 시도 실패"}

    # 다수결 투표
    vote_count = Counter(candidate_answers)
    final_answer = vote_count.most_common(1)[0][0]
    
    return {
        "answer": final_answer,
        "candidates": candidate_answers,
        "reasoning": f"총 {len(candidate_answers)}회 성공 중 최다 득표 채택"
    }

# 6. 메인 실행 로직
def run_final_step():
    results = []
    correct_count = 0
    
    # 데이터 로드
    try:
        with open("data/answer_key.jsonl", "r", encoding="utf-8") as f:
            answers = {json.loads(line)["id"]: json.loads(line)["expected_answer"] for line in f}
        with open("data/dataset.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        return

    print(f"🚀 Step 4: Self-Consistency (n=5) 시작")
    print("-" * 50)

    for line in lines:
        data = json.loads(line)
        qid = data["id"]
        question = data["question"]
        expected = str(answers.get(qid, "")).replace(" ", "").strip()
        
        print(f"[{qid}] 처리 중...", end=" ", flush=True)
        
        # 호출 전 딜레이 (안정적인 API 사용을 위함)
        time.sleep(2) 
        
        res = get_self_consistency_response(question, n=5)
        predicted = res["answer"]
        
        is_correct = (predicted == expected)
        if is_correct:
            correct_count += 1
            print(" ✅")
        else:
            print(f" ❌ (예측: {predicted} | 정답: {expected})")
            print(f"    └ 투표 내역: {res['candidates']}")

        results.append({
            "id": qid,
            "is_correct": is_correct,
            "predicted": predicted,
            "expected": expected,
            "candidates": res["candidates"]
        })

    # 최종 리포트
    accuracy = (correct_count / len(lines)) * 100
    print("-" * 50)
    print(f"📊 최종 정답률: {accuracy:.2f}% ({correct_count}/{len(lines)})")

    with open("results/results_step4.json", "w", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    run_final_step()