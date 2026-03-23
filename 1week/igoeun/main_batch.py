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

# 1. 개별 티켓 분석 스키마 (v6 튜닝 내용 적용 + ticket_id 추가)
class TicketAnalysis(BaseModel):
    ticket_id: str
    intent: Literal["order_change", "shipping_issue", "payment_issue", "refund_exchange", "other"] = Field(
        description="Category. 'other' for custom requests(gift wrap) or ambiguous context."
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="high: payment/delivery errors, unresolved past issues. medium: standard tasks (incl. ANY refund/exchange). low: general info."
    )
    needs_clarification: bool = Field(
        description="True IF core issue is hidden, action is undecided, OR custom request (gift wrap). False for simple policy questions."
    )
    route_to: Literal["order_ops", "shipping_ops", "billing_ops", "returns_ops", "human_support"] = Field(
        description="Target dept. human_support if intent='other'. returns_ops for undecided return/exchange."
    )

# 2. 여러 개의 분석 결과를 담는 최상위 리스트 스키마
class BatchClassification(BaseModel):
    results: list[TicketAnalysis]

# prompt.txt 파일 불러오기
def load_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"{file_path} 프롬프트 파일 찾기 실패")
        return None

print("- - - 데이터 배치(Batch) 분류 시작 - - -\n")

# 전체 작업 시작 시간 측정
total_start_time = time.perf_counter()

# 프롬프트 설정
prompt_path = BASE_DIR / "prompts" / "prompt_eng_v6.txt"
current_prompt = load_text_file(prompt_path)

if not current_prompt:
    print("프롬프트 불러오기 실패. 경로를 확인해주세요.")
    exit()

version_name = "batch_eng_v6"
print(f"현재 프롬프트 버전: {version_name} \n")

# 누적 변수 초기화
cumulative_input_tokens = 0
cumulative_output_tokens = 0
total_result = []
all_tickets = []

# dataset.jsonl 읽어서 리스트에 담기
with open("dataset.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue
        data = json.loads(line)
        inquiry = data.get("customer_message")
        ticket_id = data.get("id")
        
        if inquiry and ticket_id:
            all_tickets.append({"ticket_id": ticket_id, "message": inquiry})

# 3. 5개씩 묶어서 API 호출 (Batch 처리)
batch_size = 5

for i in range(0, len(all_tickets), batch_size):
    batch = all_tickets[i : i + batch_size]
    batch_input_str = json.dumps(batch, ensure_ascii=False, indent=2)
    
    # 개별 배치 시작 시간 측정
    batch_start_time = time.perf_counter()
    
    try: 
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=f"다음 고객 문의 목록을 분석하세요:\n{batch_input_str}",
            config={
                "system_instruction": current_prompt,
                "response_mime_type": "application/json",
                "response_schema": BatchClassification, 
                "temperature": 0.1,
                "max_output_tokens": 1024, # 여러 개를 출력하므로 넉넉하게 설정
            },
        )

        # JSON 파싱
        parsing_result = BatchClassification.model_validate_json(response.text)

        # 토큰 사용량 계산
        used_tokens = response.usage_metadata
        if used_tokens:
            input_tokens = used_tokens.prompt_token_count
            output_tokens = used_tokens.candidates_token_count
            cumulative_input_tokens += input_tokens
            cumulative_output_tokens += output_tokens
        else:
            input_tokens, output_tokens = 0, 0

        # 개별 배치 소요 시간 측정
        batch_end_time = time.perf_counter()
        batch_elapsed = batch_end_time - batch_start_time

        print(f"[{i+1} ~ {i+len(batch)}] 일괄 분류 완료 | 소요 시간: {batch_elapsed:.2f} 초 | 토큰(입력:{input_tokens}, 출력:{output_tokens})")

        # 결과를 total_result에 병합
        for item in parsing_result.results:
            total_result.append({
                "ticket_id": item.ticket_id,
                "analysis": item.model_dump()
            })

    except Exception as e:
        print(f"일괄 처리 중 에러 발생 (인덱스 {i}~{i+len(batch)-1}) : {e}")
        
# JSON 형식으로 파일 저장
file_name = BASE_DIR / f"classification_results_{version_name}.json"
with open(file_name, "w", encoding="utf-8") as outfile:
    json.dump(total_result, outfile, ensure_ascii=False, indent=2)

# 전체 종료 시간 기록 및 통계 출력
total_end_time = time.perf_counter()
elapsed_time = total_end_time - total_start_time

print("\n- - - - - 완료 - - - - - -")
print(f"총 처리 건수: {len(total_result)} 건")
print(f"총 소요 시간: {elapsed_time:.2f} 초")
print(f"총 입력 토큰: {cumulative_input_tokens:,} 개")
print(f"총 출력 토큰: {cumulative_output_tokens:,} 개")

estimated_cost_usd = (cumulative_input_tokens / 1_000_000 * 0.075) + (cumulative_output_tokens / 1_000_000 * 0.3)
print(f"예상 API 청구 비용: 약 ${estimated_cost_usd:.5f}")
print("- - - - - - - - - - - - - -")