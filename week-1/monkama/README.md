1. 사용한 모델, SDK, 실행 환경
- 모델 (Model): gemini-2.5-flash
- SDK: google-genai (Python)
- 실행 환경 (Environment): Windows 11 / Python 3.13 / VSCode

2. 실제 요청 구조 설명
시스템의 판단 정확도를 높이기 위해 다음과 같은 구조로 요청을 구성했습니다.

system 메시지: Pydantic의 TicketClassification 스키마 설명을 통해 각 필드(intent, urgency, needs_clarification, route_to)의 비즈니스 정의를 전달했습니다.

user 메시지: batch_data (ID와 고객 메시지 원문)를 JSON 형태로 전달했습니다.

temperature: 0.0 (일관된 분류 결과를 위해 최저값 설정)
max_tokens (max_output_tokens): 5000 (배치 처리 결과를 충분히 수용할 수 있는 크기)

3. 실제 응답 구조 설명
모델은 정의된 스키마에 따라 엄격한 JSON 구조로 응답합니다.

{
  "intent": "refund_exchange",
  "urgency": "medium",
  "needs_clarification": true,
  "route_to": "returns_ops",
  "clarification_note": "사용자가 교환과 환불 중 결정을 내리지 못한 상태이며, 상세 사유 확인이 필요함."
  "usage_tokens": 64
}

usage/token 정보:

Prompt Tokens: 약 680 ~ 720 tokens (데이터셋 크기에 따라 변동)
Candidates Tokens: 약 80 ~ 120 tokens
Total Tokens: 약 800 tokens 내외

# Prompt v1 (초기 버전)
prompt = f"""
Classify the following customer support tickets (in Korean) into the defined JSON schema.

[Classification Guide & Examples]
1. "사이즈/색상 바꿔주세요" 또는 "배송지 변경할게요"
   -> intent: order_change | urgency: medium
2. "환불/교환 절차나 가능 여부 알려주세요"
   -> intent: refund_exchange | urgency: medium
3. "처리 지연되는데 언제 되나요? 답답하네요"
   -> intent: other | urgency: high

Input Data:
{json.dumps(batch_data, ensure_ascii=False)} 
"""

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

v1 -> v2에서 바꾼 점

요약:
1. 예시 -> 일반적인 규칙으로 만듦
2. 단순히 안다는 출발에서 안다 모른다라는 로직을 삽입후 관련 규칙을 따로 더 만들어 전달
3. urgent에 대한 방파제를 세워둚으로서 티켓 SLA를 다소 높이고자 함.

- 단순 예시만 들어서는 파싱 성공률을 높일 수 없다는 것을 깨달음. 문제 상황 인지를 예시로 드는게 좋다고 생각했으나, 단지 그 예시에 해당하는 그 부분에 한정해서는 올바른 판단을 내릴 수는 있겠으나, 그외 상황들에 대해서는 확실히 AI가 멋대로 판단하는 경향이 있는 듯해서 예시를 드는 설명에서 벗어나 좀 더 일반적인 설명으로 바꿀려고 노력함.
그런데, 프롬프트2의 첫번째 규칙은 다소 구체적이긴 하나, 이거는 이것 나름대로 넒은 범용성을 가진다고 생각을 해서 추가함. 애초에 지난주에 받았던 티켓이면, 다소 기간이 지났다고 판단하고 이에대한 긴급성을 High로 두고, AI로 자동 처리하기에는 다소 복잡 난해하기에 사람으로 처리하는게 이체에 맞다고 생각을 함.
-> 그러나, 특정 규칙의 다소 넓은 범용성이 다른 규칙과의 논리적 이치에 맞지 않아 충돌이 날 수 있다고 봄. 예를 들어 첫번쨰 규칙에는 사람으로 보내라고 해놓고선, 두번쨰 규칙에는 그 의도에 맞는 부서에 보내라고 하니 AI 입장에서는 다소 난해한 입장이었을 거임. 그로인해 의도대로 안 나오고 계속 틀린 답변을 반복함. 따라서 범용성 규칙이 무조건 맞다고 하는게 아닌, 규칙 생성시 다른 규칙성과 논리적 정합성을 따져보고 검증해야함.

- 그전까지는 intent, urgency, route_to 이것만 따져보았음. 다만, 고객의 요청을 정확히 알고서 판단을 내린건지, 아니면 단순히 그렇다고 아는 척해서 판단한거지 처음에는 needs_clarification을 제대로 따져보지 않았음. 그래서 코드 로직상에 needs_clarification 라는 항목도 따져보게하고, 그렇게 true 라면 이 이유까지도 말하게끔 로직을 변경하였으며, 이와 관련된 프롬프트도 추가해두었음. 단순히 안다는 가정하에서 출발한 로직이 안다 모른다라는 추가 로직을 삽입하여 최대한 할루시네이션을 방지할려고 노력해볼려함. 그래서 규칙 중에 If the intent is multi-the intent or decision-pending, the needs_clarification=true and set the urgency to medium. 라는 로직을 추가한 이유가 그 이유임.

- urgent 에 대한 판단은 사람에게도 좀 애매한 영역이라고 판단함. 그래서 초반에는 AI에게 맞겼더니, 최소 2개 이상은 기대값과 다른 결과를 보임.그래도 일반적으로 볼떄 이건 low는 아니다, 혹은 이건 high를 둬야한다는 방파제를 세워둠. 예를 들어, if the intent is not other, Don't set the urgency to low. 라는 룰을 만든 이유는 other 이외 다른 명시된 의도들은 모두 일단 기본적으로 긴급도가 다소 있어보임. 즉, 단순 문의가 아닌 사측 입장에서 처리해야할 문제로 보였음. 그래서 other 가 아닌 intent인 경우, low로 분류하지 않도록 하도록 하였음. 그리고 - If it's regarding the request from last week,, set the urgency to high and the route_to to human_support. 라는 룰은 지난주에 처리해야했던 건들인데, 애초에 처리가 안된 요청이었다면, 기다린 고객들의 분노나 불만이 높아졌을 거라 생각하여 High로 두는게 맞다고 판단하여 이 규칙만 따로 다소 일반적이지 않게 약간 구체성있는 규칙으로 만들어 보았음.

5. 결과 비교
JSON 파싱 성공률: 약 75% ~ 83% -> 100%
exact match 개수: 9개 ~ 10개 -> 12개(전부 매치 성공)
총사용 토큰 : 약 3900 -> 약 6775
대표 실패 3건과 원인: 
- 티켓 12번: 교환 및 환불이라는 항목이 보여서 refund_exchange로 보내면 되겠지라는 판단한 거 같음. 분류상에는 맞으나, 고객이 교환과 환불 중 어떤 것을 원하는지 결정하지 못하는 상황 또한 일단, 애매모호한 문의가 맞으며, 이에 대해 정확한 처리 방향이 못 정해진 상황은 맞음. 이는 AI가 단지 분류만하고 끝내는 상황에서 사측에서 어떻게 대응할지에 대한 고민은 하지 않은 것으로 추정됨. 즉, 분류만하면 끝난다라고 판단한 것. 
- 티켓 8번, 11번: urgency를 low로 함. 그 이유가 선물용으로 포장이 가능할까요? 라는 핀트로 인해 단순 문의로 오인하고 low로 설정한 것으로 파악.
8번 긴급도 사유: "urgency_reason": "고객이 상품의 환불 가능 여부에 대해 문의하고 있으며, 즉각적인 금전적 리스크나 긴급한 처리가 필요한 상황은 아닙니다.",
11번 긴급도 사유: "urgency_reason": "고객이 선물 포장 가능 여부와 희망 배송일을 문의하고 있으며, 즉각적인 긴급 처리가 필요한 상황은 아닙니다."
- If the intent is ambiguous or unclear or multi-the intent or decision-pending, the needs_clarification=true and set the urgency to medium.
위 규칙으로 인해, 명확하지 않거나, 애매하거나 두개의 요청 사항 등이 있으면 medium으로 설정한 것으로 보임.