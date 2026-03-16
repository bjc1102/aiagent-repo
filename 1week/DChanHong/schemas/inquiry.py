from enum import Enum
from pydantic import BaseModel, Field

# 1. 고정된 선택지들을 Enum으로 정의
class IntentEnum(str, Enum):
    order_change = "order_change"
    shipping_issue = "shipping_issue"
    payment_issue = "payment_issue"
    refund_exchange = "refund_exchange"
    other = "other"

class UrgencyEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class RouteEnum(str, Enum):
    order_ops = "order_ops"
    shipping_ops = "shipping_ops"
    billing_ops = "billing_ops"
    returns_ops = "returns_ops"
    human_support = "human_support"

# 2. 메인 스키마 정의
class InquiryAnalysis(BaseModel):
    """고객 문의 분석 결과 스키마"""
    
    intent: IntentEnum = Field(
        ..., 
        description="문의 의도: 주문수정(order_change), 배송이슈(shipping_issue), 결제(payment_issue), 환불/교환(refund_exchange), 기타(other)"
    )
    
    urgency: UrgencyEnum = Field(
        ..., 
        description="긴급도: 일반(low), 우선처리(medium), 즉시확인시급(high)"
    )
    
    needs_clarification: bool = Field(
        ..., 
        description="추가 정보 필요 여부: 텍스트만으로 판단이 어려우면 true, 충분하면 false"
    )
    
    route_to: RouteEnum = Field(
        ..., 
        description="담당 부서: 주문(order_ops), 배송(shipping_ops), 결제(billing_ops), 반품(returns_ops), 수동지원(human_support)"
    )

# 예시 데이터 검증 테스트
test_data = {
    "intent": "shipping_issue",
    "urgency": "high",
    "needs_clarification": False,
    "route_to": "shipping_ops"
}

analysis = InquiryAnalysis(**test_data)
print(analysis.model_dump_json(indent=2))