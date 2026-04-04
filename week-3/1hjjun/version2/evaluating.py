import os
import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 환경 설정
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY를 .env 파일에서 찾을 수 없습니다.")
else:
    os.environ["GOOGLE_API_KEY"] = api_key

def run_golden_test():
    # 2. 인덱스 로드
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    index_path = "medical_upstage_index"
    
    if not os.path.exists(index_path):
        print(f"❌ '{index_path}' 폴더가 없습니다.")
        return

    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. 모델 설정
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    당신은 의료급여제도 전문가입니다. 
    반드시 제공된 [문맥] 정보만을 근거로 질문에 답하세요. 

    [문맥]
    {context}

    질문: {question}
    답변:""")

    # 4. 데이터셋 로드 (배열 형식 대응)
    golden_path = "golden_dataset2.jsonl"
    if not os.path.exists(golden_path):
        print(f"❌ {golden_path} 파일이 없습니다.")
        return

    print(f"\n🚀 [Golden Test] 평가 시작\n" + "="*60)

    try:
        with open(golden_path, "r", encoding="utf-8") as f:
            # 파일 전체를 읽어서 리스트로 파싱합니다.
            golden_list = json.load(f)
            
        for data in golden_list:
            q_id = data.get("id", "N/A")
            question = data.get("question")
            expected = data.get("expected_answer")

            if not question: continue

            # 5. RAG 실행
            docs = retriever.invoke(question)
            context = "\n\n".join([f"[출처: {d.metadata.get('page', '알 수 없음')}쪽] {d.page_content}" for d in docs])
            
            chain = prompt | llm | StrOutputParser()
            ai_answer = chain.invoke({"context": context, "question": question})

            print(f"🆔 ID: {q_id}")
            print(f"📝 질문: {question}")
            print(f"✅ 기대 정답: {expected}")
            print(f"🤖 AI 답변  : {ai_answer}")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        print("💡 팁: 만약 파일이 JSONL(줄바꿈 기준) 형식이라면 json.load(f) 대신 이전에 드린 코드를 사용하세요.")

if __name__ == "__main__":
    run_golden_test()