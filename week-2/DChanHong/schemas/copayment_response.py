from pydantic import BaseModel, Field, ConfigDict


class CopaymentResponse(BaseModel):
    """과제 측정용 구조화 출력 (TASK.md 권장 형식)."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        ...,
        description="본인부담률 또는 본인부담금 등 최종 답만 간결하게",
    )
    reason: str = Field(
        ...,
        description="참조 데이터 근거 또는 단계별 추론 요약",
    )
