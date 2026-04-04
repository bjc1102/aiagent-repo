"""
RAG (Retrieval-Augmented Generation) 서비스 모듈
"""

import json
from typing import Optional, Dict, Any, List

from src.core.retriever import get_relevant_documents
from src.core.llm import get_llm
from src.core.prompts import build_rag_system_prompt
from src.schemas.copayment_response import CopaymentResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ask_question(question: str) -> CopaymentResponse:
    """
    질문에 대해 검색 기반으로 답변을 생성하며, Pydantic 모델로 출력을 강제합니다.
    """
    logger.info(f"질문에 대한 RAG 답변 생성 시도: {question}")

    # 1. 문서 검색 (중요: 예외 규정 확보를 위해 k=15 확보)
    documents = get_relevant_documents(question, k=15)
    context_text = "\n".join([doc.page_content for doc in documents])
    
    # 2. 시스템 프롬프트 구성
    system_prompt = build_rag_system_prompt(context_text)
    
    # 3. LLM 호출 (Pydantic 강제)
    llm = get_llm(temperature=0.1)
    # with_structured_output를 사용하여 Pydantic 모델로 직접 응답 유도
    structured_llm = llm.with_structured_output(CopaymentResponse)
    
    from langchain_core.messages import SystemMessage, HumanMessage
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"질문: {question}")
    ]
    
    try:
        # LLM이 Pydantic 객체를 직접 반환합니다.
        response = structured_llm.invoke(messages)
        return response
        
    except Exception as e:
        logger.error(f"구조화된 출력 생성 중 오류 발생: {e}")
        # 오류 발생 시 기본값 반환 (또는 예외 처리)
        return CopaymentResponse(
            question=question,
            expected_answer="답변 생성 실패",
            source_section="N/A",
            evidence_text=f"오류: {str(e)}",
            conditions=[]
        )
