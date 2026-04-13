from pydantic import BaseModel, Field, ConfigDict

from typing import List

class CopaymentResponse(BaseModel):
    """Golden Dataset 구조에 맞춘 구조화된 출력 (TASK 3 연계)."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        description="최종 질문 내용",
    )
    expected_answer: str = Field(
        ...,
        description="본인부담률 또는 금액 등 최종 정답 요약",
    )
    source_section: str = Field(
        ...,
        description="정답 근거가 있는 PDF 문서의 섹션/표 이름",
    )
    evidence_text: str = Field(
        ...,
        description="정답을 도출한 원문의 핵심 문장/조건 리터럴",
    )
    conditions: List[str] = Field(
        ...,
        description="정답을 판단하기 위해 사용된 조건들 리스트",
    )
