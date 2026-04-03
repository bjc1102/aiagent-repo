import os
import json
import time
from google import genai
from pydantic import Field, BaseModel
from typing import Literal
from pathlib import Path


# 현재 파일 디렉터리 절대 경로 고정
BASE_DIR = Path(__file__).parent


api_key = os.getenv("GOOGLE_API_KEY") 
client = genai.Client(api_key=api_key)


# Pydantic 스키마 정의 (출력 구조 강제)
class ClassifyTicket(BaseModel) : 
    """Classify customer inquiry based on the provided schema"""
    intent: Literal["order_change", "shipping_issue", "payment_issue", "refund_exchange", "other"] = Field(
        description="Category. MUST USE 'other' IF referring to a past request without specific details, customer is unsure of the issue, or custom requests (gift wrap)."
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="high: ACTUAL payment errors, double charges, misdelivery, delayed past requests. medium: ALL refund/exchange questions, address changes, unclear app errors. low: purely general info."
    )
    needs_clarification: bool = Field(
        description="True IF customer is undecided (e.g., 'refund or exchange?'), exact issue is unknown, context missing, or custom request. False IF customer has a clear goal."
    )
    route_to: Literal["order_ops", "shipping_ops", "billing_ops", "returns_ops", "human_support"] = Field(
        description="Target dept. MUST USE 'human_support' IF intent='other'. USE 'returns_ops' for ALL refund/exchange questions."
    )


# prompt.txt 파일 불러오기
def load_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"{file_path} 프롬프트 파일 찾기 실패")
        return None


print("- - - 데이터 분류 시작 - - -\n")

# 분류 시간 측정
total_start_time = time.perf_counter()


#----------------------------------------------------
# 현재 프롬프트 설정 (프롬프트가 길어서 따로 .txt로 저장)
prompt_path = BASE_DIR / "prompts" / "prompt_eng_v6.txt"
current_prompt = load_text_file(prompt_path)

# if not current_prompt:
#     print("프롬프트 불러오기 실패, 경로 재확인 필요.")
#     exit()

# JSON 파일명에 붙을 버전명 작성
version_name = "eng_v6"
#----------------------------------------------------

print(f"현재 프롬프트 버전: prompt_{version_name} \n")


# 누적 토큰을 저장할 변수 초기화
cumulative_input_tokens = 0
cumulative_output_tokens = 0


#dataset.json 불러오고, customer_message 분류 실행
total_result = []
with open("dataset.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        if not line.strip(): # 빈 줄 건너뛰기 ㅂㄷㅂㄷㅂㄷ....
            continue

        data = json.loads(line)
        inquiry = data.get("customer_message")
        ticket_id = data.get("id")

        if not inquiry:
            continue

        try: 
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=inquiry,
                config={
                    "system_instruction": current_prompt,
                    "response_mime_type": "application/json",
                    "response_json_schema": ClassifyTicket.model_json_schema(),
                    "temperature": 0.1,
                    "max_output_tokens": 100,
                },
            )


            #JSON으로 파싱하기
            parsing_result = ClassifyTicket.model_validate_json(response.text)


            # 토큰 사용량 확인하기
            used_tokens = response.usage_metadata
            if used_tokens:
                input_tokens = used_tokens.prompt_token_count
                output_tokens = used_tokens.candidates_token_count
                cumulative_input_tokens += input_tokens
                cumulative_output_tokens += output_tokens
            else:
                input_tokens, output_tokens =0, 0

            # 분류 과정 표시
            print(f"분류 완료: {ticket_id} | 사용 토큰 - 입력: {input_tokens}, 출력: {output_tokens}")


            #결과 보기 좋게 데이터 저장
            entry_result = {
                "ticket_id": ticket_id,
                "original_message": inquiry,
                "analysis": parsing_result.model_dump()
            }
            total_result.append(entry_result)

            # 병목 현상 줄이기 (클라이언트 측 속도 제한 구현)
            time.sleep(1.5)

        except Exception as e:
            print(f"({ticket_id}) 에러 발생 : {e} -> 통과하고 다음 진행 ")
            

#JSON 형식으로 파일 저장
file_name = BASE_DIR / f"classification_results_{version_name}.json"

with open(file_name, "w", encoding="utf-8") as outfile:
    json.dump(total_result, outfile, ensure_ascii=False, indent=2)


# 파일 저장 완료 후 종료 시간 기록
total_end_time = time.perf_counter()

# 총 소요 시간 계산 (종료 시간 - 시작 시간)
elapsed_time = total_end_time - total_start_time


print("- - - - - 완료 - - - - - -\n")

print(f"총 소요 시간: {elapsed_time:.2f} 초")

print(f"총 입력 토큰: {cumulative_input_tokens:,} 개")
print(f"총 출력 토큰: {cumulative_output_tokens:,} 개")

# flash lite 모델 대략 비용 계산
estimated_cost_usd = (cumulative_input_tokens / 1_000_000 * 0.075) + (cumulative_output_tokens / 1_000_000 * 0.3)
print(f"예상 API 청구 비용: 약 ${estimated_cost_usd:.5f}")
print("- - - - - - - - - - - - - -")
