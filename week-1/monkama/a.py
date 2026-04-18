import json
from google import genai
from pydantic import BaseModel, Field
from typing import Literal, Optional

# 1. Pydantic 스키마 정의 (초압축 영문 가이드 반영)
class TicketClassification(BaseModel):
  ticket_id: str
  
  intent: Literal["order_change", "shipping_issue", "payment_issue", "refund_exchange", "other"] = Field(
    description=(
      "order_change: 주문 수정, 취소, 주소 변경, 옵션 변경 | "
      "shipping_issue: 출고, 배송 지연, 배송 누락, 배송 완료 오표시 | "
      "payment_issue: 결제 실패, 중복 결제, 청구 이상 | "
      "refund_exchange: 반품, 환불, 교환, 불량 접수 | "
      "other: 위로 단정하기 어렵거나 맥락이 부족한 경우"
    )
  )
  
  urgency: Literal["low", "medium", "high"] = Field(
    description=(
      "low: 일반 문의, 즉시 장애 아님 | "
      "medium: 처리가 필요하지만 긴급 장애/금전 리스크는 아님 | "
      "high: 결제 이상, 분실/오배송, 고객 불만 고조, 수동 확인이 시급함"
    )
  )
  urgency_reason: str = Field(
        description="해당 긴급도로 판단한 구체적인 근거를 한국어로 설명하세요."
    )
  
  needs_clarification: bool = Field(
    description=(
      "true: 현재 텍스트만으로 intent 또는 처리 방향을 단정하기 어려움 | "
      "false: 현재 정보만으로 1차 분류 가능"
    )
  )
  
  route_to: Literal["order_ops", "shipping_ops", "billing_ops", "returns_ops", "human_support"] = Field(
    description=(
      "order_ops: 주문/수정 담당 | "
      "shipping_ops: 배송 담당 | "
      "billing_ops: 결제/청구 담당 | "
      "returns_ops: 환불/교환 담당 | "
      "human_support: 맥락 부족, 다부서 이슈, 에스컬레이션 필요"
    )
  )
  
  clarification_note: Optional[str] = Field(
    default=None,
    description="needs_clarification이 true일 때만 한국어로 사유를 설명하세요."
  )

class TicketClassificationList(BaseModel):
  results: list[TicketClassification]

# 2. 클라이언트 세팅 및 데이터 로드
client = genai.Client() # 환경 변수 GEMINI_API_KEY 설정 필요
file_path = "dataset.jsonl"

batch_data = []
expected_outputs = {} 

with open(file_path, "r", encoding="utf-8") as file:
  for line in file:
    if not line.strip(): continue
    data = json.loads(line)
    batch_data.append({
      "id": data["id"],
      "msg": data["Cmsg"]
    })
    if "expected_output" in data:
      expected_outputs[data["id"]] = data["expected_output"]

# 3. 최적화된 비즈니스 로직 프롬프트 (v2 반영)
# # Prompt v2 (개선 버전)
prompt = f"""
Classify the following customer support tickets (in Korean) into the defined JSON schema based on these mandatory rules:

[Classification Guide]
- If it's regarding the request from last week,, set the urgency to high and the route_to to human_support. 

- IF the intent is not "order", Don't set the route_to to human_support.

- If the intent is clear or single, the needs_clarification=false. 

- If the intent is ambiguous or unclear or multi-the intent or decision-pending, the needs_clarification=true and set the urgency to medium.

- if the intent is not other, Don't set the urgency to low.

- if the intent is other, route_to must be the human_support and the needs_clarification must be true.


Input Data:
{json.dumps(batch_data, ensure_ascii=False, separators=(',', ':'))} 
"""

# 4. 모델 호출 및 결과 처리
try:
  response = client.models.generate_content(
    model="gemini-2.5-flash", # 또는 gemini-2.5-flash
    contents=prompt,
    config={
      'response_mime_type': 'application/json',
      'response_schema': TicketClassificationList,
      'temperature': 0.0,
      'max_output_tokens': 8192
    }
  )
  
  validated_result = TicketClassificationList.model_validate_json(response.text)
  usage = response.usage_metadata
  total_out_tokens = usage.candidates_token_count if usage else 0
  total_chars = sum(len(res.model_dump_json()) for res in validated_result.results)
  
  print("\n" + "="*60)
  for result in validated_result.results:
    print(f"\n--- [ID: {result.ticket_id}] ---")
    
    # 정답지와 대조 검증
    expected = expected_outputs.get(result.ticket_id)
    if expected:
        # 1. 비교할 필드명들을 모델 속성명과 똑같이 맞춥니다.
        check_keys = ["intent", "urgency", "route_to", "needs_clarification"]
        mismatches = []

        for k in check_keys:
            predicted_val = getattr(result, k) # 모델이 예측한 값
            expected_val = expected.get(k)     # 정답지에 적힌 값

            # 정답지에 해당 항목이 있고, 예측값과 다를 경우에만 기록
            if expected_val is not None and predicted_val != expected_val:
                mismatches.append(f"[{k}] 예측:{predicted_val} ↔ 정답:{expected_val}")
        
        if not mismatches:
            print("✅ 모든 항목 매칭 성공! (분류/라우팅/모호성 모두 일치)")
        else:
            print(f"❌ 매칭 실패! ({', '.join(mismatches)})")
        # ------------------------
    
    # 개별 토큰량 추산 및 출력
    my_chars = len(result.model_dump_json())
    estimated_tokens = int(total_out_tokens * (my_chars / total_chars)) if total_chars > 0 else 0
    
    final_output = result.model_dump(exclude={"ticket_id"})
    final_output["usage_tokens"] = estimated_tokens
    print(json.dumps(final_output, indent=2, ensure_ascii=False))

  print("\n" + "="*60)
  print(f"📊 총 사용 토큰: {usage.total_token_count if usage else 'N/A'}")
  print("="*60 + "\n")

except Exception as e:
  print(f"\nAPI 처리 중 오류 발생: {e}")
