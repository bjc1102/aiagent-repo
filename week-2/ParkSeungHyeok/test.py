#!/usr/bin/env python3

import json
import base64
import os, re
from collections import Counter
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import fitz

load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"), 
    base_url=os.getenv("BASE_URL") # 구글 서버로 라우팅
)


class CopaymentResult(BaseModel):
    reason: str
    answer: str

# --- 파일 읽기 헬퍼 함수 ---
def get_pdf_text(pdf_path):
    print(f"📄 PDF 텍스트 추출 중: {pdf_path}")
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def get_image_base64(image_path):
    print(f"🖼️ 이미지 인코딩 중: {image_path}")
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# 데이터 로드 헬퍼 함수
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# ==========================================
# 파이프라인 엔진: 각 Step별 테스트 로직
# ==========================================

def run_evaluation(step_name, dataset, answer_key, context, technique="zero-shot"):
    print(f"\n🚀 [실행 중] {step_name}")
    
    correct_count = 0
    total_count = len(dataset)
    
    for i in range(total_count):
        q_data = dataset[i]
        true_answer = str(answer_key[i]["expected_answer"])
        question = q_data["question"]
        
        # 기법에 따른 시스템 프롬프트 분기 처리
        if technique == "zero-shot":
            sys_prompt = f"""아래는 의료급여 본인부담률 데이터입니다. 질문에 대해 정확한 본인부담률을 답하세요 
            [중요 지시사항]
            최종 정답(answer)은 구구절절 설명하지 말고, 반드시 정답지 포맷(예: '5%', '무료', '60,000원', '해당되지 않음')과 같이 '단답형'으로만 출력하세요..\n{context}"""
            temp = 0.0
            
        elif technique == "few-shot":
            sys_prompt = f"""아래는 의료급여 본인부담률 데이터입니다.
            [중요 지시사항]
            최종 정답(answer)은 구구절절 설명하지 말고, 반드시 정답지 포맷(예: '5%', '무료', '60,000원', '해당되지 않음')과 같이 '단답형'으로만 출력하세요.
            {context}

            [예시 1]
            Q: 1종 수급권자, 65세 이상, 틀니
            A: 5%
            [예시 2]
            Q: 2종 수급권자, 15세 이하, 입원
            A: 3%
            """
            temp = 0.0
            
        elif technique == "cot" or technique == "self-consistency":
            sys_prompt = f"""당신은 본인부담률 산정 전문가입니다. 아래 데이터를 바탕으로 답변하세요.
            [중요 지시사항]
            최종 정답(answer)은 구구절절 설명하지 말고, 반드시 정답지 포맷(예: '5%', '무료', '60,000원', '해당되지 않음')과 같이 '단답형'으로만 출력하세요.
            {context}

            반드시 다음 단계를 거쳐 생각(reason)하고, 최종 정답(answer)을 도출하세요:
            1. 수급권자의 종별(1종/2종)을 확인합니다.
            2. 나이 및 특정 질환(만성질환, 장애 등) 조건을 확인합니다.
            3. 의료기관 종별(1차/2차/3차)을 확인하여 최종 비율을 계산합니다.

            [중요 지시사항]
            최종 정답(answer)은 구구절절 설명하지 말고, 반드시 정답지 포맷(예: '5%', '무료', '60,000원', '해당되지 않음')과 같이 '단답형'으로만 출력하세요.
            """
            # self-consistency는 창의성(다양한 경로)을 위해 온도를 높임
            temp = 0.7 if technique == "self-consistency" else 0.0

        # --- AI 호출 (Self-Consistency vs 일반) ---
        if technique == "self-consistency":
            answers = []
            # 동일 프롬프트를 5번 반복 실행하여 앙상블(다수결) 효과 생성
            for _ in range(5):
                response = client.beta.chat.completions.parse(
                    model="gemini-2.5-flash",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": question}
                    ],
                    response_format=CopaymentResult,
                    temperature=temp
                )
                answers.append(response.choices[0].message.parsed.answer)
            
            # 가장 많이 나온 정답을 최종 정답으로 채택
            ai_final_answer = Counter(answers).most_common(1)[0][0]
            ai_reason = f"다수결 투표 결과: {answers}"
            
        else:
            # 일반 1회 호출
            response = client.beta.chat.completions.parse(
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question}
                ],
                response_format=CopaymentResult, # Pydantic 구조 강제
                temperature=temp
            )
            result = response.choices[0].message.parsed
            ai_final_answer = result.answer
            ai_reason = result.reason

        # --- 채점 로직 ---
        # 1. 숫자를 추출해서 비교 (예: "72000원" -> 72000, "72,000원" -> 72000)
        ai_digits = re.sub(r'[^0-9]', '', ai_final_answer)
        true_digits = re.sub(r'[^0-9]', '', true_answer)
        
        # 2. 똑똑한 정답 판별기
        if "무료" in true_answer and ("무료" in ai_final_answer or "0%" in ai_final_answer):
            is_correct = True
        elif true_digits != "" and ai_digits == true_digits:
            is_correct = True
        elif true_answer in ai_final_answer:
            is_correct = True
        else:
            is_correct = False

        if is_correct:
            correct_count += 1
            
        print(f"[{i+1}/{total_count}] 정답여부: {'✅' if is_correct else '❌'} | AI답: {ai_final_answer} (실제답: {true_answer})")
        if not is_correct:
             print(f"   ㄴ AI 추론과정: {ai_reason}")

    accuracy = (correct_count / total_count) * 100
    print(f"📊 {step_name} 최종 정답률: {accuracy:.1f}% ({correct_count}/{total_count})")
    return accuracy

def main():
    pdf_text = get_pdf_text("../image/2024 알기 쉬운 의료급여제도.pdf")
    image_base64 = get_image_base64("../image/image.png")
    
    context_data = f"[의료급여제도 안내서(PDF) 내용]\n{pdf_text}"



    dataset = load_jsonl("../data/dataset.jsonl")
    answer_key = load_jsonl("../data/answer_key.jsonl")
    
    # 3. Few-shot 예시 제외 처리 (과제 요구사항)
    # 앞의 3개 항목을 예시로 썼다고 가정하고, 평가 데이터에서 잘라냅니다.
    eval_dataset = dataset[3:]
    eval_answer_key = answer_key[3:]

    print("=== 프롬프트 엔지니어링 성능 평가 시작 ===")
    
    # Step 1: Zero-shot
    run_evaluation("Step 1 (Zero-shot)", eval_dataset, eval_answer_key, context_data, "zero-shot")
    
    # Step 2: Few-shot
    run_evaluation("Step 2 (Few-shot)", eval_dataset, eval_answer_key, context_data, "few-shot")
    
    # Step 3: CoT
    run_evaluation("Step 3 (CoT)", eval_dataset, eval_answer_key, context_data, "cot")
    
    # Step 4: Self-Consistency
    run_evaluation("Step 4 (Self-Consistency)", eval_dataset, eval_answer_key, context_data, "self-consistency")

if __name__ == "__main__":
    main()