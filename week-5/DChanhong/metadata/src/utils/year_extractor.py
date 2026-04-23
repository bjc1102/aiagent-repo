"""
질문 문장에서 "어느 년도 자료를 검색해야 하는가" 를 추출한다.
검색 단계의 Pre-retrieval filter 에 쓰기 위한 유틸.

전략:
  1) 명시적 년도 (정규식 "202\d")
  2) 상대 표현 ("작년"·"올해"·"내년") → REFERENCE_YEAR 기준으로 환산
  3) 비교 키워드 ("대비"·"변화"·"달라진" 등) → Cross-year 플래그 on
     → 감지된 년도가 여러 개거나 cross-year 면 다중 필터 (`$in`) 적용

LLM 분류기 대신 규칙 기반을 쓰는 이유:
  - 의료급여 도메인의 질문은 패턴이 단순 (명시 년도 or 상대 표현 대부분)
  - 비용·지연 없음, 재현성 높음
"""

import re
from typing import List, TypedDict


class YearExtractionResult(TypedDict):
    years: List[str]        # 필터에 사용할 source_year 값들 (e.g. ["2025"], ["2025","2026"])
    is_cross_year: bool     # 여러 년도 비교성 질문인지
    rationale: str          # 디버그용 설명


class YearExtractor:
    # 비교·변화를 나타내는 키워드
    CROSS_YEAR_KEYWORDS = (
        "대비", "비교", "변화", "달라진", "차이",
        "이전", "신설", "개정", "변동", "증감", "인상", "인하",
    )

    # 상대 년도 표현 → 기준년도 대비 offset
    RELATIVE_YEAR_KEYWORDS = {
        "작년": -1,
        "지난해": -1,
        "전년": -1,
        "올해": 0,
        "이번 해": 0,
        "금년": 0,
        "당해": 0,
        "내년": 1,
        "다음 해": 1,
        "신년": 1,
    }

    def __init__(self, reference_year: int):
        self.reference_year = reference_year

    def extract(self, question: str) -> YearExtractionResult:
        # 1) 명시적 년도 (202x)
        explicit_years = set(re.findall(r"(202\d)", question))

        # 2) 상대 표현
        relative_years = set()
        matched_kw = []
        for kw, offset in self.RELATIVE_YEAR_KEYWORDS.items():
            if kw in question:
                relative_years.add(str(self.reference_year + offset))
                matched_kw.append(kw)

        years = sorted(explicit_years | relative_years)

        # 3) Cross-year 플래그
        is_cross_year = any(kw in question for kw in self.CROSS_YEAR_KEYWORDS)
        # 년도가 둘 이상 잡혔으면 자동으로 cross-year 로 본다
        if len(years) > 1:
            is_cross_year = True

        rationale_parts = []
        if explicit_years:
            rationale_parts.append(f"explicit={sorted(explicit_years)}")
        if matched_kw:
            rationale_parts.append(f"relative={matched_kw}")
        if is_cross_year:
            rationale_parts.append("cross_year=True")
        rationale = "; ".join(rationale_parts) or "no_year_signal"

        return {
            "years": years,
            "is_cross_year": is_cross_year,
            "rationale": rationale,
        }
