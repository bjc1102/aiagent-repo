import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

def format_docs_with_metadata(docs):
    """컨텍스트에 년도와 페이지 정보를 명시적으로 노출"""
    formatted = []
    for doc in docs:
        meta = doc.metadata
        header = f"[출처: {meta['source_year']}년도 지침서 / 섹션: {meta['section_title']} / {meta['page']}쪽]"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n" + "-"*50 + "\n\n".join(formatted)

# 1. 인덱스 로드
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.load_local(
    "medical_advanced_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 2. Retriever 및 LLM 설정
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # 소주제 단위이므로 k를 약간 줄여도 정보량이 충분합니다.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 3. 프롬프트 구성
template = """당신은 의료급여제도 전문가입니다. 아래 제공된 컨텍스트를 바탕으로 사용자의 질문에 정확하게 답변하세요.

지침:
1. 각 컨텍스트 상단에 표기된 [출처 년도]를 반드시 확인하세요.
2. 질문에서 특정 연도(예: 2026년)를 언급했다면, 해당 연도의 정보만 사용하여 답변하세요.
3. 두 연도의 정보가 컨텍스트에 모두 포함되어 있고 수치가 다르다면, "2025년에는 X였으나, 2026년에는 Y로 변경되었습니다"와 같이 비교하여 설명하세요.
4. 컨텍스트에 답변 근거가 없다면 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 정직하게 답하세요.

컨텍스트:
{context}

질문: {question}

답변:"""

rag_prompt = ChatPromptTemplate.from_template(template)

# 4. RAG 체인 구축
rag_chain = (
    {"context": retriever | format_docs_with_metadata, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
)

if __name__ == "__main__":
    query = "2025년과 비교해서 2026년에 달라진 1종 수급권자의 외래 본인부담금 기준을 알려줘."
    print(f"\n🔍 질문: {query}")
    print("="*50)
    
    result = rag_chain.invoke(query)
    print(f"\n💡 AI 답변:\n{result.content}")