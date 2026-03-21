from typing import Literal
from pydantic import BaseModel

class TicketOutput(BaseModel):
    intent: Literal[
        "order_change",
        "shipping_issue",
        "payment_issue",
        "refund_exchange",
        "other"
    ]
    urgency: Literal["low", "medium", "high"]
    needs_clarification: bool
    route_to: Literal[
        "order_ops",
        "shipping_ops",
        "billing_ops",
        "returns_ops",
        "human_support"
    ]


    # 모델이 반환해야 하는 JSON 구조를 코드로 정의하는 파일. 예를 들면, intent, urgency 값이 허용된 것인지 검증할 때 씀