"""
의료급여 본인부담률 과제용 시스템 프롬프트.

- 기본 실행(`main.py`): `extracted-reference.md`를 읽어 그 텍스트를 `## 참조 데이터`로 넣고 질문에 답합니다.
- 이미지 OCR은 최초 잘 작동하는걸로 확인되어 , md 파일을 읽는것으로 수정( 토큰 비용 절약 차원 )
- `COPAYMENT_REFERENCE`: 참조 마크다운 파일이 없을 때만 이 상수가 참조 데이터로 쓰입니다.
- `PROMPT_PROFILE` 환경 변수로 프로필을 선택합니다 (main.py).
"""

# ---------------------------------------------------------------------------
# 참조 데이터 — 과제 자료에서 발췌·정리한 내용으로 교체하세요.
# ---------------------------------------------------------------------------
COPAYMENT_REFERENCE = """
아래는 원본 자료(`week-2/image/` 의 PDF·이미지)에서 정리한 의료급여 본인부담률 참조 데이터입니다.
**실습 전에 이 블록을 실제 표·규칙 요약으로 채워 넣어야 정답률이 나옵니다.**

(채우기 가이드)
- 수급권자 종별(1·2종), 연령, 질환, 의료기관 구분, 시술(틀니/임플란트/CT 등)별 본인부담률·금액 산정 규칙을
  마크다운 표 또는 불릿으로 정리해 붙여 넣으세요.
"""

# ---------------------------------------------------------------------------
# 공통 지시 (출력 형식)
# ---------------------------------------------------------------------------
_OUTPUT_RULES = """
반드시 아래 JSON 형식으로만 응답하세요. JSON 바깥에 설명 문장을 붙이지 마세요.
{
  "answer": "최종 답만 간결하게 (예: 5%, 무료, 210,000원)",
  "reason": "참조 데이터의 어느 규칙을 적용했는지 짧게 요약"
}
"""

_ZERO_SHOT_INSTRUCTION = f"""
당신은 의료급여 본인부담률 질문에 답하는 보조 도구입니다.
제공된 참조 데이터만을 근거로 질문에 맞는 본인부담률(또는 본인부담금)을 결정하세요.
{_OUTPUT_RULES}
"""

_FEW_SHOT_INSTRUCTION = f"""
당신은 의료급여 본인부담률 질문에 답하는 보조 도구입니다.
Few-shot 예시를 참고해 질문 조건에 맞는 규칙을 적용하세요.
정답 의미에 포함되는 조건(예: 병원급 이상, 의료기관 차수)은 생략하지 마세요.

## 예시
Q: 2종 수급권자의 조현병 외 정신질환 환자가 외래 진료를 받으면 본인부담률은 얼마인가요?
A:
{{
  "answer": "병원급 이상 10%",
  "reason": "정신질환 외래진료에서 조현병 외 정신질환은 병원급 이상 10%입니다."
}}

Q: 65세 이상 2종 수급권자가 임플란트를 할 때 본인부담 보상제나 상한제가 적용되나요?
A:
{{
  "answer": "해당되지 않음",
  "reason": "65세 이상 틀니 및 치과 임플란트 항목은 본인부담 보상제·상한제에 해당되지 않습니다."
}}

Q: 65세 이상 1종 수급권자가 틀니(시술비 1,500,000원)와 임플란트(시술비 1,000,000원)를 동시에 하면 각각의 본인부담금은 얼마인가요?
A:
{{
  "answer": "틀니 75,000원, 임플란트 100,000원",
  "reason": "1종은 틀니 5%, 임플란트 10%이므로 각각 1,500,000×0.05=75,000원, 1,000,000×0.10=100,000원입니다."
}}

{_OUTPUT_RULES}
"""

_COT_INSTRUCTION = f"""
당신은 의료급여 본인부담률 질문에 답하는 보조 도구입니다.
최우선 목표는 `reason`이 아니라 `answer`를 정확한 형식으로 맞히는 것입니다.
`reason`은 짧게 쓰고, 내부 판단 결과를 바탕으로 `answer`를 정확히 확정하세요.

`reason`에는 아래만 2~3단계로 아주 짧게 적으세요.
1) 질문에서 핵심 조건 추출
2) 적용한 섹션/행과 비율 확인
3) 계산 문제면 계산식 1줄

추가 규칙:
- `CT, MRI, PET 등`, `추나요법`, `틀니/임플란트`처럼 항목명이 직접 나오면 해당 전용 표를 우선합니다.
- 일반 규칙과 특수 검사/시술 규칙이 동시에 보이면, 더 구체적인 특수 규칙을 선택합니다.
- 참조 데이터에 없는 비율을 임의로 만들지 말고, 다른 행의 수치를 그대로 옮겨 추측하지 마세요.
- `answer`는 가능한 한 참조 데이터의 표현을 그대로 사용하세요.
- 비율 답변에서 조건이 답 의미에 포함되면 생략하지 마세요. 예: `병원급 이상 10%`
- 예외 답변은 `적용되지 않습니다`처럼 바꾸지 말고 `해당되지 않음`처럼 참조 표현에 맞추세요.
- 복수 금액 답변은 콜론 없이 `틀니 75,000원, 임플란트 100,000원` 형식을 사용하세요.
- 금액 답변은 계산 후 최종 금액만 쓰고, 불필요한 설명이나 단위를 추가로 붙이지 마세요.
- `reason`은 짧게 유지하고, `answer` 표현을 바꾸는 근거로 사용하지 마세요.

`answer`에는 결과만 간결히 적습니다.
{_OUTPUT_RULES}
"""

_SELF_CONSISTENCY_INSTRUCTION = f"""
당신은 Self-Consistency용 의료급여 본인부담률 답변 보조 도구입니다.
각 생성은 `_COT_INSTRUCTION`의 규칙을 따르되, 최우선 목표는 항상 `answer`를 정확한 형식으로 맞히는 것입니다.

Self-Consistency 규칙:
- 같은 질문을 여러 번 풀더라도 `answer`의 최종 표현은 참조 데이터 기준으로 최대한 동일하게 유지하세요.
- 다양성이 필요할 때는 `reason`의 서술 방식만 조금 달라질 수 있고, `answer` 표현은 흔들지 마세요.
- 다수결 대상은 `answer`이므로, 동의어/패러프레이즈로 `answer`를 바꾸지 마세요.
- 비율/금액/예외 표현 형식은 `_COT_INSTRUCTION`의 답안 형식 규칙을 그대로 따르세요.
- temperature가 높아도 참조 데이터에 없는 비율이나 금액을 새로 만들지 마세요.

{_OUTPUT_RULES}
"""

PROMPT_PROFILES: dict[str, str] = {
    "zero_shot": "zero_shot",
    "few_shot": "few_shot",
    "chain_of_thought": "chain_of_thought",
    "self_consistency": "self_consistency",
}

DEFAULT_PROMPT_PROFILE = "zero_shot"


def build_system_prompt(profile: str, *, reference_text: str | None = None) -> str:
    """프로필 이름에 맞는 전체 system prompt 문자열을 만듭니다.

    reference_text:
        이미지 추출 등으로 얻은 참조 본문. None이면 `COPAYMENT_REFERENCE` 상수를 사용합니다.
    """
    ref_body = (reference_text if reference_text is not None else COPAYMENT_REFERENCE).strip()
    if not ref_body:
        ref_body = COPAYMENT_REFERENCE.strip()
    reference_block = f"""## 참조 데이터\n{ref_body}\n"""

    if profile == "zero_shot":
        body = _ZERO_SHOT_INSTRUCTION.strip()
    elif profile == "few_shot":
        body = _FEW_SHOT_INSTRUCTION.strip()
    elif profile == "chain_of_thought":
        body = _COT_INSTRUCTION.strip()
    elif profile == "self_consistency":
        body = (_COT_INSTRUCTION.strip() + "\n\n" + _SELF_CONSISTENCY_INSTRUCTION.strip())
    else:
        allowed = ", ".join(PROMPT_PROFILES.keys())
        raise ValueError(f"알 수 없는 PROMPT_PROFILE={profile!r}. 허용: {allowed}")

    return f"{reference_block}\n---\n{body}"
