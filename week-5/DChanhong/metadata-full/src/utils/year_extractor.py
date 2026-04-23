"""
질문 문장에서 "어느 년도 자료를 검색해야 하는가" 를 추출한다.
검색 단계의 Pre-retrieval filter 에 쓰기 위한 유틸.

metadata/ 버전과 동일 로직.
"""

import re
from typing import List, TypedDict


class YearExtractionResult(TypedDict):
    years: List[str]
    is_cross_year: bool
    rationale: str


class YearExtractor:
    CROSS_YEAR_KEYWORDS = (
        "대비", "비교", "변화", "달라진", "차이",
        "이전", "신설", "개정", "변동", "증감", "인상", "인하",
    )

    RELATIVE_YEAR_KEYWORDS = {
        "작년": -1, "지난해": -1, "전년": -1,
        "올해": 0, "이번 해": 0, "금년": 0, "당해": 0,
        "내년": 1, "다음 해": 1, "신년": 1,
    }

    def __init__(self, reference_year: int):
        self.reference_year = reference_year

    def extract(self, question: str) -> YearExtractionResult:
        explicit_years = set(re.findall(r"(202\d)", question))

        relative_years = set()
        matched_kw = []
        for kw, offset in self.RELATIVE_YEAR_KEYWORDS.items():
            if kw in question:
                relative_years.add(str(self.reference_year + offset))
                matched_kw.append(kw)

        years = sorted(explicit_years | relative_years)

        is_cross_year = any(kw in question for kw in self.CROSS_YEAR_KEYWORDS)
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
