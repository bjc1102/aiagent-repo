#  1주차 과제: LLM 호출을 테스트하고 이해하기

---

## 1. 사용한 모델, SDK, 실행 환경
- Model: gemini-3.1-flash-lite-preview
- SDK: Python용 Google GenAI SDK
- Library:
    - GenAI
    - Python: 3.11 (가상환경 이용)
    - Pydantic
- 실행 환경:
    - Windows (VScode)
    - 윈도우 환경 변수에 API KEY 저장

## 2. 실제 요청 구조 설명

### 2.1 system message

prompt_v2.txt

당신은 10년 차 베테랑 고객 센터 CS 매니저입니다. 주어진 고객 문의를 분석하여 알맞은 카테고리로 분류하세요. 아래의 분류 기준과 예외 규칙을 엄격하게 적용해야 합니다.

[분류 기준]
1. intent (문의 유형)
   - order_change: 주문 수정, 취소, 주소 변경, 옵션 변경
   - shipping_issue: 출고 확인, 배송 지연, 배송 누락, 배송 완료 오표시
   - payment_issue: 결제 실패, 중복 결제, 청구 이상
   - refund_exchange: 반품, 환불, 교환, 불량 접수
   - other: 시스템상 기본 제공되지 않는 요청(예: 선물 포장), 또는 맥락이 부족하여 위 4가지로 단정하기 어려운 경우

2. urgency (긴급도)
   - low: 일반 문의, 즉시 장애 아님 (예: 절차 문의, 정책 질문)
   - medium: 처리가 필요하지만 긴급 장애/금전 리스크는 아님 (예: 배송지 변경, 교환/환불 접수, 일반적인 배송 조회)
   - high: 결제 이상, 분실/오배송, 고객 불만 고조, 수동 확인이 시급함 (예: 결제 오류, 중복 결제, 배송 완료 오표시, 장기 미처리 건)

3. needs_clarification (추가 확인 필요 여부)
   - true: 현재 텍스트만으로 intent 또는 처리 방향을 단정하기 어려움
   - false:현재 정보만으로 1차 분류 가능

4. route_to (담당 부서)
   - order_ops: 주문/수정 담당
   - shipping_ops: 배송 담당
   - billing_ops: 결제/청구 담당
   - returns_ops: 환불/교환 담당
   - human_support: 다부서 이슈, 에스컬레이션 필요, intent가 'other'이거나 needs_clarification이 'true'인 복합적인 경우

[주의 및 예외 규칙 (Edge Cases)]
- "선물 포장" 같은 시스템 외적인 요구는 옵션 변경(order_change)이 아니라 기타(other) 및 human_support로 분류
- 고객이 "교환할지 환불할지 고민 중"처럼 명확한 액션을 정하지 않았다면 needs_clarification을 true로 설정하되, 문의의 성격이 명확하다면(예: 반품/교환) human_support가 아닌 해당 담당 부서(returns_ops 등)로 배정하세요
- 배송지 변경"이나 "단순 환불 요청"은 일상적인 CS 업무이므로 urgency를 medium으로 분류하세요
- 고객이 단순히 "결제 문제인지 모르겠다"며 원인을 묻는 일반적인 앱 오류 상황은, 실제 결제 실패나 중복 결제가 확인되지 않은 이상 urgency를 medium으로 분류하세요

### 2.2 user 메시지
- `dataset.jsonl` 파일의 `customer_message` 파라미터 참조
- `main.py`의 `inquiry`

### 2.3 model
- gemini-3.1-flash-lite-preview
- Google AI Studio의 비율 제한에서 확인 결과, 무료 티어에서 RPM/TPM/RPD 조건을 충족하는 모델을 선택했습니다.

### 2.4 temperature
- `temperature`:0.1
    - JSON 형식 출력은 Pydantic을 활용하여 고정했기 때문에, 기본 Gemini API 권장 설정인 0.1을 설정했습니다.

### 2.5 max_tokens
-`max_output_token`: 450
    - 토큰 수를 계산한 결과 한 라인당 36개의 토큰이 소비되었습니다.
    - `max_output_token`을 400으로 설정한 결과 오류가 발생했습니다.
    - 단순하게 36*12를 해서 432를 넣어주었지만 분류가 진행되지 않고 계속 대기하는 현상이 발생했습니다.
    - 토큰 수를 450으로 설정하여 실행 결과 무리없이 잘 출력이 되어서 일단 450으로 설정하였습니다.....

## 3. 실제 응답 구조 설명
### 3.1 본문 JSON 예시
```
  {
    "ticket_id": "ticket-01",
    "original_message": "주문한 러닝화가 아직 도착하지 않았어요. 배송이 어디까지 왔는지 확인하고 싶습니다.",
    "analysis": {
      "intent": "shipping_issue",
      "urgency": "medium",
      "needs_clarification": false,
      "route_to": "shipping_ops"
    }
  },

```

## 4. v1 -> v2에서 바꾼점

모호한 문의 사항에 대해서 Edge Cases를 추가했습니다.

[주의 및 예외 규칙 (Edge Cases)]
- "선물 포장" 같은 시스템 외적인 요구는 옵션 변경(order_change)이 아니라 기타(other) 및 human_support로 분류
- 고객이 "교환할지 환불할지 고민 중"처럼 명확한 액션을 정하지 않았다면 needs_clarification을 true로 설정하되, 문의의 성격이 명확하다면(예: 반품/교환) human_support가 아닌 해당 담당 부서(returns_ops 등)로 배정하세요
- 배송지 변경"이나 "단순 환불 요청"은 일상적인 CS 업무이므로 urgency를 medium으로 분류하세요
- 고객이 단순히 "결제 문제인지 모르겠다"며 원인을 묻는 일반적인 앱 오류 상황은, 실제 결제 실패나 중복 결제가 확인되지 않은 이상 urgency를 medium으로 분류하세요

## 5. 결과 비교
- JSON 파싱 성공률: 100%
- exact macth 개수
    - v1: 7
    - v2: 12

### 5.1 대표 실패 사레

#### 5.4.1 긴급도(urgency)가 기댓값과 다르게 설정된 경우

-ticket-05
    - 기댓값은 `medium`인데 AI는 `high`로 분류하였습니다.
    - 출고 전 수정 요청을 수동 확인이 시급함이라고 판단한 것으로 추정됩니다.

- ticket-08
    - 기댓값은 `medium`인데 AI는 `low`로 분류하였습니다.
    - 고객 문의사항의 실제 의도인 환불보다는 `먼저 알고 싶다`를 중심으로 두어 정책 문의로 판단한 것 같습니다.

- ticket-12
    - 기댓값은 `medium`인데 AI는 `low`로 분류하였습니다.
    - ticket-08 케이스와 마찬가지로 교환이나 환불보다는 `어떤 절차로 진행하면 될까요?`에 초점을 맞춰 판단한 것으로 추정됩니다.


```
  {
    "ticket_id": "ticket-05",
    "original_message": "배송지를 잘못 적었는데 아직 출고 전이면 수정하고 싶습니다.",
    "analysis": {
      "intent": "order_change",
      "urgency": "high",
      "needs_clarification": false,
      "route_to": "order_ops"
    }
  },
                        ⁝
  {
    "ticket_id": "ticket-08",
    "original_message": "포장은 안 뜯었는데 환불이 가능한지 먼저 알고 싶습니다.",
    "analysis": {
      "intent": "refund_exchange",
      "urgency": "low",
      "needs_clarification": false,
      "route_to": "returns_ops"
    }
  },
                        ⁝
  {
    "ticket_id": "ticket-12",
    "original_message": "사이즈가 안 맞아서 교환할지 환불할지 고민 중인데 어떤 절차로 진행하면 될까요?",
    "analysis": {
      "intent": "refund_exchange",
      "urgency": "low",
      "needs_clarification": false,
      "route_to": "returns_ops"
    }
  }
```

- v2 프롬프트에 다음과 같은 문구를 추가하여 조정하였습니다.
    - ticket-05: "배송지 변경"이나 "단순 환불 요청"은 일상적인 CS 업무이므로 urgency를 medium으로 분류하세요
    - ticket-08/12: 고객이 단순히 "결제 문제인지 모르겠다"며 원인을 묻는 일반적인 앱 오류 상황은, 실제 결제 실패나 중복 결제가 확인되지 않은 이상 urgency를 medium으로 분류하세요.

```
  {
    "ticket_id": "ticket-05",
    "original_message": "배송지를 잘못 적었는데 아직 출고 전이면 수정하고 싶습니다.",
    "analysis": {
      "intent": "order_change",
      "urgency": "medium",
      "needs_clarification": false,
      "route_to": "order_ops"
    }
  },
                        ⁝
  {
    "ticket_id": "ticket-08",
    "original_message": "포장은 안 뜯었는데 환불이 가능한지 먼저 알고 싶습니다.",
    "analysis": {
      "intent": "refund_exchange",
      "urgency": "medium",
      "needs_clarification": false,
      "route_to": "returns_ops"
    }
  },
                        ⁝
  {
    "ticket_id": "ticket-12",
    "original_message": "사이즈가 안 맞아서 교환할지 환불할지 고민 중인데 어떤 절차로 진행하면 될까요?",
    "analysis": {
      "intent": "refund_exchange",
      "urgency": "medium",
      "needs_clarification": true,
      "route_to": "returns_ops"
    }
  }
```

#### 5.4.2  ticket 11 - intent와 route_to 분류 실패

기댓값은 `other/human_support`인데 AI는 `order_change/order_ops`로 분류했습니다.
AI는 선물 포장 요청을 옵션 변경으로 판단한 것 같습니다.

```
 {
    "ticket_id": "ticket-11",
    "original_message": "선물용으로 포장 가능한가요? 가능하면 이번 주 안에 받고 싶어요.",
    "analysis": {
      "intent": "order_change",
      "urgency": "medium",
      "needs_clarification": false,
      "route_to": "order_ops"
    }
  }
```

다음과 같은 로직을 v2 프롬프트에 추가하였습니다.
    - "선물 포장" 같은 시스템 외적인 요구는 옵션 변경(order_change)이 아니라 기타(other) 및 human_support로 분류

```
  {
    "ticket_id": "ticket-11",
    "original_message": "선물용으로 포장 가능한가요? 가능하면 이번 주 안에 받고 싶어요.",
    "analysis": {
      "intent": "other",
      "urgency": "medium",
      "needs_clarification": false,
      "route_to": "human_support"
    }
  },
```

#### 5.4.3 ticket 12 - urgency/needs_clarification 분류 실패

기댓값은 `needs_clarification: true`인데 AI는 `false`로 분류했습니다.
고객이 행동을 결정하지 않은 상태를 AI가 포착하지 못한것 같습니다.

```
{
    "ticket_id": "ticket-12",
    "original_message": "사이즈가 안 맞아서 교환할지 환불할지 고민 중인데 어떤 절차로 진행하면 될까요?",
    "analysis": {
      "intent": "refund_exchange",
      "urgency": "low",
      "needs_clarification": false,
      "route_to": "returns_ops"
    }
}
```
다음과 같은 문구를 추가하였습니다. 
- 고객이 "교환할지 환불할지 고민 중"처럼 명확한 액션을 정하지 않았다면 needs_clarification을 true로 설정하되, 문의의 성격이 명확하다면(예: 반품/교환) human_support가 아닌 해당 담당 부서(returns_ops 등)로 배정하세요.

```
  {
    "ticket_id": "ticket-12",
    "original_message": "사이즈가 안 맞아서 교환할지 환불할지 고민 중인데 어떤 절차로 진행하면 될까요?",
    "analysis": {
      "intent": "refund_exchange",
      "urgency": "medium",
      "needs_clarification": true,
      "route_to": "returns_ops"
    }
  }
```

---

초기 구현 이후, **비용 절감(Token Diet)**과 **처리 속도 향상(Batch Processing)**을 목표로 코드 최적화를 시도하였습니다.

---

# 6. 토큰 사용량 최적화

LLM 호출 시 발생하는 API 비용을 최소화하기 위해 프롬프트 다이어트와 구조화된 출력(Structured Output)의 특징을 활용했습니다.

## 6.1 문제점
- 높은 입력 토큰: 초기 자연어(한글) 위주의 긴 지시문(v3)으로 인해 1회 호출당 약 680개의 입력 토큰이 소모되었습니다.
- 낮은 토큰 효율성: 텍스트 생성형 모델과 달리, 분류/추출 태스크에서는 서술형 프롬프트가 불필요한 비용을 발생시켰습니다.

## 6.2 개선 과정
1. 영문 프롬프트 번역: LLM의 토큰화(Tokenization) 특성을 고려하여 프롬프트를 영문으로 변경, 즉각적인 토큰 절감 효과를 얻었습니다.
2. Pydantic Schema description 활용: 시스템 프롬프트에 있던 예외 규칙(Edge Cases)을 모두 지우고, Pydantic 모델 내 필드의 description 속성으로 프롬프트를 축약하여 작성하였습니다.
3. 토큰을 줄임과 동시에 모호한 문맥(과거 요청건, 절차 문의 등)에서 정확도가 떨어지는 문제를 해결하기 위해 MUST USE, ACTUAL, CONFIRMED 등의 제약 키워드를 추가하였습니다.


## 6.3 프롬프트 진화 과정 및 비교 (Prompt Evolution)
최적화 과정에서 프롬프트와 스키마를 여러 차례 수정하며 겪은 트레이드오프(Trade-off)와 해결 과정입니다.

### 1. v1~v3 (초기)

**prompt_v3 기준**

- 장황한 한글 지시문, CS 매니저 페르소나 부여
- 텍스트가 길어 토큰 낭비가 심함
- 영어 프롬프트 실행 시, 입력 토큰 사용량이 감소함

```
- - - prompt_v3.txt - - -

현재 프롬프트 버전: prompt_V3

분류 완료: ticket-01 | 사용 토큰 - 입력: 450, 출력: 44
분류 완료: ticket-02 | 사용 토큰 - 입력: 449, 출력: 44
분류 완료: ticket-03 | 사용 토큰 - 입력: 451, 출력: 44
분류 완료: ticket-04 | 사용 토큰 - 입력: 449, 출력: 44
분류 완료: ticket-05 | 사용 토큰 - 입력: 441, 출력: 44
분류 완료: ticket-06 | 사용 토큰 - 입력: 444, 출력: 44
분류 완료: ticket-07 | 사용 토큰 - 입력: 444, 출력: 44
분류 완료: ticket-08 | 사용 토큰 - 입력: 441, 출력: 44
분류 완료: ticket-09 | 사용 토큰 - 입력: 446, 출력: 42
분류 완료: ticket-10 | 사용 토큰 - 입력: 441, 출력: 42
분류 완료: ticket-11 | 사용 토큰 - 입력: 444, 출력: 42
분류 완료: ticket-12 | 사용 토큰 - 입력: 451, 출력: 44
- - - - - 완료 - - - - - -

총 입력 토큰: 5,351 개
총 출력 토큰: 522 개
예상 API 청구 비용: 약 $0.00056
- - - - - - - - - - - - - -
```

#### 실행 결과

```
- - - prompt_eng_v3.txt - - -

현재 프롬프트 버전: prompt_eng_v3

분류 완료: ticket-01 | 사용 토큰 - 입력: 401, 출력: 44
분류 완료: ticket-02 | 사용 토큰 - 입력: 400, 출력: 44
분류 완료: ticket-03 | 사용 토큰 - 입력: 402, 출력: 44
분류 완료: ticket-04 | 사용 토큰 - 입력: 400, 출력: 44
분류 완료: ticket-05 | 사용 토큰 - 입력: 392, 출력: 44
분류 완료: ticket-06 | 사용 토큰 - 입력: 395, 출력: 44
분류 완료: ticket-07 | 사용 토큰 - 입력: 395, 출력: 44
분류 완료: ticket-08 | 사용 토큰 - 입력: 392, 출력: 44
분류 완료: ticket-09 | 사용 토큰 - 입력: 397, 출력: 42
분류 완료: ticket-10 | 사용 토큰 - 입력: 392, 출력: 42
분류 완료: ticket-11 | 사용 토큰 - 입력: 395, 출력: 42
분류 완료: ticket-12 | 사용 토큰 - 입력: 402, 출력: 44

- - - - - 완료 - - - - - -

총 입력 토큰: 4,763 개
총 출력 토큰: 522 개
예상 API 청구 비용: 약 $0.00051

- - - - - 정확도 - - - - - -

평가 프롬프트: classification_results_eng_v3.json

전체 항목: 12개
정답 항목: 12개
정확도: 100.0%

```

### 2. v4~v5 (압축/영문 프롬프트 사용)

- 영문 번역, 핵심 예외 규칙(Edge Cases)만 프롬프트에 남김
- 간결한 분류 작업이라 페르소나 삭제
- 한글 대비 토큰은 크게 절감되었으나 여전히 프롬프트 의존도가 높음

#### 실행 결과

```
- - - prompt_eng_v5.txt - - -

현재 프롬프트 버전: prompt_eng_v5

분류 완료: ticket-01 | 사용 토큰 - 입력: 273, 출력: 44
분류 완료: ticket-02 | 사용 토큰 - 입력: 272, 출력: 44
분류 완료: ticket-03 | 사용 토큰 - 입력: 274, 출력: 44
분류 완료: ticket-04 | 사용 토큰 - 입력: 272, 출력: 44
분류 완료: ticket-05 | 사용 토큰 - 입력: 264, 출력: 44
분류 완료: ticket-06 | 사용 토큰 - 입력: 267, 출력: 44
분류 완료: ticket-07 | 사용 토큰 - 입력: 267, 출력: 44
분류 완료: ticket-08 | 사용 토큰 - 입력: 264, 출력: 44
분류 완료: ticket-09 | 사용 토큰 - 입력: 269, 출력: 42
분류 완료: ticket-10 | 사용 토큰 - 입력: 264, 출력: 42
분류 완료: ticket-11 | 사용 토큰 - 입력: 267, 출력: 42
분류 완료: ticket-12 | 사용 토큰 - 입력: 274, 출력: 44
- - - - - 완료 - - - - - -

총 입력 토큰: 3,227 개
총 출력 토큰: 522 개
예상 API 청구 비용: 약 $0.00040

- - - - - 정확도 - - - - - -

평가 프롬프트: classification_results_eng_v5.json

전체 항목: 12개
정답 항목: 12개
정확도: 100.0%

모든 분류가 정답과 일치합니다.

```


### 3. v6 (프롬프트 -> 스키마 description으로 이동)

- 프롬프트는 1줄로 축소, 모든 규칙을 스키마 description으로 이동
- 총 사용 토큰:
- 정확도: 83.3%
- 토큰 사용량은 크게 감소했으나, 모델이 경계(Boundary) 데이터를 오판하기 시작함

#### 프롬프트
```
Classify the inquiry. Strictly apply the edge cases defined in the schema descriptions.
```

#### schema description

```
class ClassifyTicket(BaseModel) : 
    """Classify customer inquiry based on the provided schema"""
    intent: Literal["order_change", "shipping_issue", "payment_issue", "refund_exchange", "other"] = Field(
        description="Category. 'other' for custom requests(gift wrap) or ambiguous context"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="high: CONFIRMED payment/delivery errors, unresolved past issues. medium: standard tasks, refunds, UNCONFIRMED app errors. low: general info."
    )
    needs_clarification: bool = Field(
        description="True IF core issue is hidden, action is undecided, OR custom request (gift wrap). False for simple policy questions."
    )
    route_to: Literal["order_ops", "shipping_ops", "billing_ops", "returns_ops", "human_support"] = Field(
        description="Target dept. human_support if intent='other'. returns_ops for undecided return/exchange"
    )
```

#### 실행 결과
```
- - - 데이터 분류 시작 - - -

현재 프롬프트 버전: prompt_eng_v6 

분류 완료: ticket-01 | 사용 토큰 - 입력: 43, 출력: 44
분류 완료: ticket-02 | 사용 토큰 - 입력: 42, 출력: 44
분류 완료: ticket-03 | 사용 토큰 - 입력: 44, 출력: 44
분류 완료: ticket-04 | 사용 토큰 - 입력: 42, 출력: 44
분류 완료: ticket-05 | 사용 토큰 - 입력: 34, 출력: 44
분류 완료: ticket-06 | 사용 토큰 - 입력: 37, 출력: 44
분류 완료: ticket-07 | 사용 토큰 - 입력: 37, 출력: 44
분류 완료: ticket-08 | 사용 토큰 - 입력: 34, 출력: 44
분류 완료: ticket-09 | 사용 토큰 - 입력: 39, 출력: 42
분류 완료: ticket-10 | 사용 토큰 - 입력: 34, 출력: 42
분류 완료: ticket-11 | 사용 토큰 - 입력: 37, 출력: 42
분류 완료: ticket-12 | 사용 토큰 - 입력: 44, 출력: 44

- - - - - 완료 - - - - - -

총 소요 시간: 39.41 초
총 입력 토큰: 467 개
총 출력 토큰: 522 개
예상 API 청구 비용: 약 $0.00019

- - - - - 정확도 - - - - - -

```

### 4. v8 (최종)

#### 프롬프트
```
Classify the inquiry. Strictly apply the edge cases defined in the schema descriptions.
```

#### schema description
```
class ClassifyTicket(BaseModel) : 
    """Classify customer inquiry based on the provided schema"""
    intent: Literal["order_change", "shipping_issue", "payment_issue", "refund_exchange", "other"] = Field(
        description="Category. MUST USE 'other' IF referring to a past request without specific details, customer is unsure of the issue, or custom requests (gift wrap)."
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="high: ACTUAL payment errors, double charges, misdelivery, delayed past requests. medium: ALL refund/exchange questions, address changes, unclear app errors. low: purely general info."
    )
    needs_clarification: bool = Field(
        description="True IF customer is undecided (e.g., 'refund or exchange?'), exact issue is unknown, context missing, or custom request. False IF customer has a clear goal."
    )
    route_to: Literal["order_ops", "shipping_ops", "billing_ops", "returns_ops", "human_support"] = Field(
        description="Target dept. MUST USE 'human_support' IF intent='other'. USE 'returns_ops' for ALL refund/exchange questions."
    )

```

- 스키마 내 강력한 제약어(MUST USE, ACTUAL) 적용으로 조건 구체화
- 총 사용 토큰: 약 130개
- 정확도: 100%
- 토큰 사용량 감소 및 목표 분류 정확도 달성

```
- - - prompt_eng_v6.txt - - -

현재 프롬프트 버전: prompt_eng_v6 

분류 완료: ticket-01 | 사용 토큰 - 입력: 43, 출력: 44
분류 완료: ticket-02 | 사용 토큰 - 입력: 42, 출력: 44
분류 완료: ticket-03 | 사용 토큰 - 입력: 44, 출력: 44
분류 완료: ticket-04 | 사용 토큰 - 입력: 42, 출력: 44
분류 완료: ticket-05 | 사용 토큰 - 입력: 34, 출력: 44
분류 완료: ticket-06 | 사용 토큰 - 입력: 37, 출력: 44
분류 완료: ticket-07 | 사용 토큰 - 입력: 37, 출력: 44
분류 완료: ticket-08 | 사용 토큰 - 입력: 34, 출력: 44
분류 완료: ticket-09 | 사용 토큰 - 입력: 39, 출력: 42
분류 완료: ticket-10 | 사용 토큰 - 입력: 34, 출력: 42
분류 완료: ticket-11 | 사용 토큰 - 입력: 37, 출력: 42
분류 완료: ticket-12 | 사용 토큰 - 입력: 44, 출력: 44

- - - - - 완료 - - - - - -

총 입력 토큰: 467 개
총 출력 토큰: 522 개
예상 API 청구 비용: 약 $0.00019

- - - - - 정확도 - - - - - -

평가 프롬프트: classification_results_eng_v6.json

전체 항목: 12개
정답 항목: 12개
정확도: 100.0%

모든 분류가 정답과 일치합니다.
```

## 6.4 결과
- 총 입력 토큰 소모량을 약 5,351개 -> 467개 수준으로 대략 90% 절감
- 프롬프트를 축약하고, 구체적인 엣지 케이스 방어 로직을 통해 정확도 100%를 달성

# 7. 속도 및 안정성 최적화 (Batch Processing)

- main.batch.py
- 데이터를 처리할 때 발생하는 API Rate Limit 병목 현상 해결을 시도하였습니다.

## 7.1 문제점
- Rate Limit 병목: 데이터를 순차적(Sequential)으로 1건씩 처리하는 for 반복문 구조에서는 API 요청이 단시간에 집중되었습니다.
- 과도한 딜레이: 이로 인해 구글 서버의 분당 요청 제한(RPM)에 걸려 특정 티켓(06, 09, 11번 등)에서 **평소의 20배에 달하는 대기 시간(Delay)**이 발생하는 현상이 관찰되었습니다.

## 7.2 개선 과정
- 일괄 처리(Batch) 로직 도입: 1건씩 API를 호출하는 방식에서, Pydantic 최상위 스키마를 List 형태로 변경(BatchClassification)하여 데이터를 5개씩 묶어서(Batch size=5) 한 번에 전송하도록 구조를 리팩토링했습니다.
- 시간 측정(Perf_counter) 추가: 파이썬 내장 모듈인 time.perf_counter()를 도입하여 배치 처리당 소요 시간 및 전체 실행 시간을 모니터링했습니다.

## 7.3 결과

```
main.py

총 소요 시간: 39.41 초
총 입력 토큰: 467 개
총 출력 토큰: 522 개
예상 API 청구 비용: 약 $0.00019
```

```
main_batch.py

[1 ~ 4] 일괄 분류 완료 | 소요 시간: 1.99 초 | 토큰(입력:229, 출력:252)
[5 ~ 8] 일괄 분류 완료 | 소요 시간: 2.76 초 | 토큰(입력:200, 출력:252)
[9 ~ 12] 일괄 분류 완료 | 소요 시간: 1.83 초 | 토큰(입력:212, 출력:246)

총 처리 건수: 12 건
총 소요 시간: 6.00 초
총 입력 토큰: 641 개
총 출력 토큰: 750 개
예상 API 청구 비용: 약 $0.00027
```

- API 호출 횟수 80% 감소 & 처리 속도 대폭 향상
  - 12건의 티켓을 처리하기 위해 12번 호출하던 것을 3번의 호출로 줄였습니다.
  - 네트워크 대기 및 재시도 로직이 사라지면서 전체 분류 소요 시간이 단축되었습니다.

- 토큰 사용량 증가
  - 총 입력과 출력 토큰 사용량이 증가하였습니다.

## 회고 및 요약
- LLM을 활용한 구조화된 데이터 추출 시, "프롬프트는 최대한 짧게, 스키마 제약 조건(Description)은 최대한 날카롭게" 작성하는 것이 정확도와 비용을 모두 잡는 핵심임을 확인하였습니다.
- 대량의 데이터를 다루는 AI 파이프라인에서는 개별 처리보다 배치(Batch) 처리 구조를 설계하는 것이 성능과 서버 안정성 측면에서 나을 수 있음을 경험했습니다. 

