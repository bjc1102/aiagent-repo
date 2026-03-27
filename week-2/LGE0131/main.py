import os
import json
import time
from pathlib import Path
from collections import Counter
from pydantic import BaseModel, Field
from google import genai
from evaluater import evaluate_predictions
from steps import zero_shot, few_shot, cot, self_consistency


# 경로 설정
BASE_DIR = Path(__file__).parent

med_aid_img = BASE_DIR / "image" / "image.png"
mid_aid_questions = BASE_DIR / "data" / "dataset.jsonl"

expected_answer_file = BASE_DIR / "data" / "answer_key.jsonl"

results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True) # 결과 폴더 자동 생성


# API 연결
api_key = os.getenv("GOOGLE_API_KEY") 
client = genai.Client(api_key=api_key)


# File API 이미지 업로드 (한 번만 올려두고 계속 재사용하는거 )
print("이미지 업로드 중...")
img_file = client.files.upload(file=med_aid_img)
print("업로드 완료!")


# 기본 응답 스키마
class BaseResponse(BaseModel):
    answer: str = Field(description="최종 본인부담률 (예: '5%', '10%', '면제')")
    reason: str = Field(description="정답을 도출한 이유")


# CoT 질문 추론 과정 구조화 스키마
class CoTResponse(BaseModel):
    beneficiary_type: str = Field(description="의료급여 1종인지 2종인지 판별")
    med_aid_facility: str = Field(description="의료기관 종류 (1차, 2차, 3차 등)")
    medical_cost_condition: str = Field(description="비용 및 기타 특이 조건 (연령 등)")
    answer: str = Field(description="최종 본인부담률")
    reason: str = Field(description="정답을 도출한 논리적 이유")


# 3. 질문 데이터 로드 
def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

questions_data = load_jsonl(mid_aid_questions)
answers_data = load_jsonl(expected_answer_file)


answer_key = {item["id"]: item["expected_answer"] for item in answers_data}


# gemini API 호출 함수
def call_gemini(prompt_text, question_text, schema_class, temperature=0.0):
    """Gemini API 호출 공통 함수"""
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt_text, img_file, question_text],
        config={
            "response_mime_type": "application/json",
            "response_schema": schema_class,
            "temperature": temperature,
        }
    )
    if not response.text:
        print("API 응답 텍스트가 비어있음. (Safety 필터링 등 의심)")
        return {"answer": "오류", "reason": "API 응답이 없습니다."}

    return json.loads(response.text)



# - - - - - - - - - - 현재 실행 단계 설정 - - - - - - - - - - - - - - 
current_step = "zero_shot"
# - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - -


# 5. 메인 실행 로직 
if __name__ == "__main__":
    print("- - - - - - - - - - - - 응답 시작 - - - - - - - - - - - -")

    start_time = time.time()

    
    # [Step 1] Zero-shot - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    step1_preds, step1_metrics = zero_shot.run(client, img_file, questions_data)

    end_time = time.time()
    exec_time = end_time - start_time

    used_tokens = {
        "prompt_tokens": step1_metrics["prompt_tokens"],
        "completion_tokens": step1_metrics["completion_tokens"],
        "total_tokens": step1_metrics["total_tokens"]
    }

    exec_time = step1_metrics["elapsed_time_sec"]

    evaluate_predictions(
        step_name=current_step, 
        predictions_data=step1_preds, 
        answer_key=answer_key, 
        results_dir=results_dir,
        execution_time=exec_time,
        token_usage=used_tokens
    )
    
    # [Step 2] Few-shot - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # step2_preds, step2_metrics = few_shot.run(client, img_file, questions_data)

    # end_time = time.time()
    # exec_time = end_time - start_time

    # used_tokens = {
    #     "prompt_tokens": step2_metrics["prompt_tokens"],
    #     "completion_tokens": step2_metrics["completion_tokens"],
    #     "total_tokens": step2_metrics["total_tokens"]
    # }

    # exec_time = step2_metrics["elapsed_time_sec"]

    # evaluate_predictions(
    #     step_name=current_step, 
    #     predictions_data=step2_preds, 
    #     answer_key=answer_key, 
    #     results_dir=results_dir,
    #     execution_time=exec_time,
    #     token_usage=used_tokens
    # )

    # [Step 3] Chain of Thought - - - - - - - - - - - - - - - - - - - - -
    # step3_preds, step3_metrics = cot.run(client, img_file, questions_data)
    
    # end_time = time.time()
    # exec_time = end_time - start_time

    # used_tokens = {
    #     "prompt_tokens": step3_metrics["prompt_tokens"],
    #     "completion_tokens": step3_metrics["completion_tokens"],
    #     "total_tokens": step3_metrics["total_tokens"]
    # }

    # exec_time = step3_metrics["elapsed_time_sec"]

    # evaluate_predictions(
    #     step_name=current_step, 
    #     predictions_data=step3_preds, 
    #     answer_key=answer_key, 
    #     results_dir=results_dir,
    #     execution_time=exec_time,
    #     token_usage=used_tokens
    # )

    # [Step 4] Self-Consistency - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # step4_preds, step4_metrics = self_consistency.run(client, img_file, questions_data)

    # end_time = time.time()
    # exec_time = end_time - start_time

    # used_tokens = {
    #     "prompt_tokens": step4_metrics["prompt_tokens"],
    #     "completion_tokens": step4_metrics["completion_tokens"],
    #     "total_tokens": step4_metrics["total_tokens"]
    # }

    # exec_time = step4_metrics["elapsed_time_sec"]

    # evaluate_predictions(
    #     step_name=current_step, 
    #     predictions_data=step4_preds, 
    #     answer_key=answer_key, 
    #     results_dir=results_dir,
    #     execution_time=exec_time,
    #     token_usage=used_tokens
    # )

    print("- - - - - - - - - - - - - - 완료 - - - - - - - - - - - - - -")