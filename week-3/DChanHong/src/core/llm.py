"""
LLM 인스턴스 생성 및 관리 모듈
Gemini (OpenAI 호환 엔드포인트) 또는 GPT 등을 활용할 수 있도록 설정합니다.
"""

from typing import Optional, Any
# Gemini 전용 라이브러리를 사용합니다.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.config.settings import (
    GEMINI_API_KEY, 
    LLM_MODEL, 
    OPENAI_API_KEY
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    **kwargs
) -> Any:
    """
    설정된 모델에 따라 적절한 Chat 모델 인스턴스를 반환합니다.
    """
    target_model = model_name or LLM_MODEL
    
    logger.info(f"LLM 인스턴스 생성 시도: {target_model}")

    # Gemini 모델인 경우 Google 전용 라이브러리 사용
    if "gemini" in target_model.lower():
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY를 찾을 수 없습니다. .env 파일을 확인하세요.")
            raise ValueError("GEMINI_API_KEY is missing.")
            
        return ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=GEMINI_API_KEY,
            temperature=temperature,
            **kwargs
        )
    
    # 그 외 (GPT 등)
    else:
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY가 없습니다. API 호출 시 오류가 발생할 수 있습니다.")
            
        return ChatOpenAI(
            model=target_model,
            openai_api_key=OPENAI_API_KEY,
            temperature=temperature,
            **kwargs
        )
