import os
from openai import OpenAI
from dotenv import load_dotenv
from schemas.inquiry import InquiryAnalysis # 정의한 스키마 불러오기

load_dotenv()

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        # 2026년 기준 최신 Flash 모델 사용 (추론 능력과 속도의 균형)
        self.model_name = "gemini-3-flash-preview" 
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY가 .env 파일에 없습니다.")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def get_config(self) -> dict:
        """현재 서비스에서 사용하는 모델 및 호출 옵션 정보를 반환합니다."""
        return {
            "model": self.model_name,
            "base_url": self.base_url,
            "response_format": {
                "type": "json_schema",
                "json_schema_name": "inquiry_analysis",
            },
            "reasoning_effort": "low",
        }

    def analyze_inquiry(self, customer_text: str) -> InquiryAnalysis:
        """고객 문의를 분석하여 구조화된 데이터를 반환합니다."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 고객 서비스 분석 전문가입니다. 주어진 문의 내용을 분석하여 정해진 JSON 규격에 맞춰 응답하세요."
                    },
                    {"role": "user", "content": customer_text}
                ],
                # [핵심 옵션 1] 구조화된 응답 강제 (Response Format)
                # Pydantic 모델의 스키마를 JSON 형태로 전달합니다.
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "inquiry_analysis",
                        "schema": InquiryAnalysis.model_json_schema()
                    }
                },
                # [핵심 옵션 2] 추론 노력 설정
                # 단순 분류 작업이므로 'low' 또는 'minimal'이 효율적입니다.
                extra_body={
                    "reasoning_effort": "low" 
                }
            )

            # AI의 응답(JSON 문자열)을 Pydantic 객체로 변환
            json_result = response.choices[0].message.content
            return InquiryAnalysis.model_validate_json(json_result)

        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            raise