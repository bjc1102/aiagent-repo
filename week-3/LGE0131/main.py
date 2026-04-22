import os
import json
import re
import getpass
from pathlib import Path
from typing import List, cast, Union
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

BASE_DIR = Path(__file__).parent

med_aid_pdf = BASE_DIR / "data" / "2024 알기 쉬운 의료급여제도.pdf"
med_aid_questions = BASE_DIR / "data" / "golden_dataset.jsonl"


# LangSmith API 연결
os.environ["LANGSMITH_TRACING"] = "false"

# OpenAI API 연결
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# 스키마 정의
class MedAidResponse(BaseModel):
    answer: Union[str, int] = Field(
        description="질문에 대한 최종 답변. 반드시 기호나 단위(%, 원 등)를 포함한 '문자열(String)' 형태로 작성하세요. 숫자만 쓰면 안 됩니다. (예: '50%', '3,000원', '무료')"
    )
    evidence_text: str = Field(description="정답을 도출하기 위해 참조한 원문의 핵심 텍스트 (그대로 발췌)")
    condition: List[str] = Field(description="정답을 판단하기 위해 필요한 전제 조건들 리스트 (예: ['의료급여 1종', '상급종합병원', '입원'])")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 메인 파이프라인
def med_aid_response():

    # 1. PDF 로드 
    loader = PDFPlumberLoader(str(med_aid_pdf))
    splits = loader.load()

    # chunk_size = 1000
    # chunk_overlap = 200
    
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    #     separators=["\n\n", "\n", " ", ""]
    # )

    # 청크 검증 
    table_chunks = [doc.page_content for doc in splits if "구분" in doc.page_content and "본인부담률" in doc.page_content]

    print(f"표 데이터 추정 청크(페이지) {len(table_chunks)}개 발견. 샘플 출력:")
    for i, chunk in enumerate(table_chunks[:2]):
        print(f"\n--- [ 샘플 청크 {i+1} ] ---")
        print(chunk[:300] + "...\n") # 너무 기니까 300자만 출력
    print("-"*50 + "\n")

    # 2. 임베딩 및 벡터 저장소 구축
    embedding_model_name = "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(model=embedding_model_name)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(BASE_DIR / "chroma_db_page_level")
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. LLM 및 체인 구성
    llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)
    structured_llm = llm.with_structured_output(MedAidResponse)

    template="""
    다음 문맥(Context)을 기반으로 질문에 답하세요. 답변은 반드시 JSON 형태로 출력해주세요.
    
    문맥:
    {context}
    
    질문: {question}
    """

    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )

    # JSON으로 결과 출력
    output = {
        "metadata": {
            "loader": "PDFPlumberLoader",
            "text_splitter": "사용 안 함 (페이지 단위 보존)",
            "chunk_count": len(splits),
            "embedding_model": embedding_model_name,
            "vector_store": "Chroma DB"
        },
        "results": []
    }

    dataset_path = str(med_aid_questions)
    golden_dataset = [json.loads(line) for line in open(dataset_path, "r", encoding="utf-8")]

    print(f"총 {len(splits)}개의 청크(페이지)로 분할되었습니다.")
    
    print("\n" + "="*50)
    print("검색 품질 확인 및 AI 답변 생성")
    print("="*50)

    success_count = 0

    for item in golden_dataset:
        q_id = item["id"]
        question = item["question"]
        expected_answer = item["expected_answer"]
        
        # evidence_text를 리스트 형태로 통일
        raw_evidence = item["evidence_text"]
        evidence_texts = raw_evidence if isinstance(raw_evidence, list) else [raw_evidence]
        
        # 1. Top-K 검색 수행 2. 정답 근거 매칭 확인
        retrieved_docs = retriever.invoke(question)
        retrieved_content_nospace = "".join([doc.page_content for doc in retrieved_docs]).replace(" ", "").replace("\n", "")
        
        is_success = False
        for evidence in evidence_texts:
            keywords = [kw.strip() for kw in evidence.replace("->", "=").split("=")]
            if all(kw.replace(" ", "") in retrieved_content_nospace for kw in keywords):
                is_success = True
                break
                
        if is_success:
            print(f"검색 성공 [{q_id}]: 정답 근거가 포함됨")
            success_count += 1
        else:
            print(f"검색 실패 [{q_id}]: Top-K 청크에 정답 근거가 없음")
        
        # 생성
        raw_result = rag_chain.invoke(question)
        ai_result = cast(MedAidResponse, raw_result)
        
        final_answer = ai_result.answer
        if isinstance(final_answer, int):
            final_answer = f"{final_answer:,}원"
        else:
            final_answer = str(final_answer)
        
        result_entry = {
            "q_id": q_id,
            "question": question,
            "answer": final_answer,
            "evidence_text": ai_result.evidence_text,
            "condition": ai_result.condition,
            "expected_answer": expected_answer
        }
        
        output["results"].append(result_entry)
        print(f"   └── AI 분석 완료")

    print("\n" + "-" * 50)
    print(f"최종 검색 성공률: 총 {len(golden_dataset)}문제 중 {success_count}문제 성공")
    print("-" * 50 + "\n")


    output_json_path = BASE_DIR / "evaluation_report_final.json"

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"최종 결과가 '{output_json_path}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    med_aid_response()