# V1 Prompt Result

## 평가 기준

- 정답 데이터: `1week/dataset.pretty.json`
- 분석 결과: `1week/V2/json/result/20260318-185847/analysis-results.json`
- 비교 필드: `intent`, `urgency`, `needs_clarification`, `route_to`

## 정확도 요약

- 완전 일치(4개 필드 모두 일치): `9/12` = `75.0%`
- 필드 단위 정확도: `45/48` = `93.8%`

### 필드별 정확도

- `intent`: `12/12` = `100%`
- `urgency`: `10/12` = `83.3%`
- `needs_clarification`: `11/12` = `91.7%`
- `route_to`: `12/12` = `100%`

## 오차 항목

### ticket-08

- 기대값: `urgency=medium`
- 예측값: `urgency=low`
- 해석: 환불 가능 여부를 묻는 문의를 너무 낮은 긴급도로 판단했다.

### ticket-09

- 기대값: `urgency=high`
- 예측값: `urgency=medium`
- 해석: 장기 미처리 불만을 높은 긴급도로 끌어올리지 못했다.

### ticket-12

- 기대값: `needs_clarification=true`
- 예측값: `needs_clarification=false`
- 해석: 교환/환불 사이에서 고민 중인 절차 문의를 명확한 요청으로 과판단했다.

## 결과 해석

- `intent`와 `route_to`는 전건 정답으로, 문의 유형 분류와 라우팅은 안정적이다.
- 주요 오차는 `urgency`와 `needs_clarification`에 집중되어 있다.
- 특히 모델이 긴급도를 한 단계 낮게 판단하는 경향이 보인다.
- 절차 상담형 문의는 실제 액션 요청과 구분하는 기준을 더 명확히 줄 필요가 있다.

## 난이도별 관찰

- `normal`: `4/6` 완전 일치
- `boundary`: `3/3` 완전 일치
- `ambiguous`: `2/3` 완전 일치

## 사용량 메모

- 총 요청 수: `12`
- 총 토큰: `9,438`
- 평균 응답 시간: `2,152.67ms`
- 총 실행 시간: `25,832.03ms`

## V2 개선 가설

- `urgency` 기준에 "지연 누적", "중복 결제/오결제", "처리 지연 불만" 같은 high 신호를 더 명시한다.
- "정보 문의/절차 상담"과 "즉시 처리 요청"을 구분하는 `needs_clarification` 기준을 강화한다.
- 개선 후에는 동일한 `dataset.pretty.json` 기준으로 완전 일치율과 필드 정확도를 다시 비교한다.
