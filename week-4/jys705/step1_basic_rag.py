import json
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 설정
CHUNKING_SIZE = 500
CHUNKING_OVERLAP = 100
DB_PATH = "./chroma_db_week4"
GOLDEN_DATA_PATH = "golden_dataset.jsonl"

def indexing():
    if os.path.exists(DB_PATH):
        print("🚀 [Step 1-1] 기존 구축된 벡터 DB를 로드합니다...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        
    print("🚀 [Step 1-1] 다년도 문서 인덱싱 시작...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    years = ["2025", "2026"]
    all_splits = []
    
    for year in years:
        file_path = f"data/{year} 알기 쉬운 의료급여제도.pdf"
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_year"] = year
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNKING_SIZE, chunk_overlap=CHUNKING_OVERLAP
        )
        splits = text_splitter.split_documents(docs)
        all_splits.extend(splits)
        print(f"✅ {year}년 문서 청킹 완료 ({len(splits)}개 청크)")

    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=embeddings, persist_directory=DB_PATH
    )
    return vectorstore

def get_rag_chain(vectorstore):
    template = """아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 출처 년도(source_year)가 포함되어 있습니다. 질문이 묻는 년도와 일치하는 정보를 우선적으로 사용하세요.
정보가 부족하면 "정보를 찾을 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def format_docs_with_year(docs):
        return "\n\n".join([f"[출처: {d.metadata['source_year']}년] {d.page_content}" for d in docs])

    chain = (
        {"context": retriever | format_docs_with_year, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain, retriever

def evaluate_golden_dataset(chain, retriever):
    print("\n📊 [Step 1-3] 골든 데이터셋 20문항 전수 평가 시작...\n")
    
    with open(GOLDEN_DATA_PATH, 'r', encoding='utf-8') as f:
        golden_data = [json.loads(line) for line in f]

    total_q = len(golden_data)
    correct_answers = 0
    chunks_included_count = 0

    for i, item in enumerate(golden_data):
        q_id = f"q{i+1:02d}"
        question = item['question']
        expected_ans = item['expected_answer']
        target_year = item['source_year']
        difficulty = item['difficulty']

        # 1. 문서 검색 
        retrieved_docs = retriever.invoke(question)
        retrieved_years = [doc.metadata.get('source_year', 'Unknown') for doc in retrieved_docs]
        
        # 검색된 청크의 모든 텍스트를 하나로 합침 (공백 제거 후 비교를 위함)
        combined_content = "".join([doc.page_content for doc in retrieved_docs]).replace(" ", "").replace("\n", "")
        
        # 2. LLM 답변 생성
        answer = chain.invoke(question).content

        # ---------------- 평가 로직 ----------------
        expected_compact = expected_ans.replace(" ", "")
        
        # [청크 포함 여부 판정] - DB에서 가져온 원문들 안에 정답이 들어있는가?
        chunk_inclusion = "O" if expected_compact in combined_content else "X"
        if chunk_inclusion == "O":
            chunks_included_count += 1

        # [최종 정답 판정] - LLM이 내뱉은 답변에 정답이 들어있는가?
        answer_compact = answer.replace(" ", "")
        is_correct_answer = expected_compact in answer_compact
        ans_mark = "정답" if is_correct_answer else "오답"
        if is_correct_answer: 
            correct_answers += 1

        # ---------------- 결과 출력 ----------------
        print(f"[{q_id}] 난이도: {difficulty} | 목표 연도: {target_year}")
        print(f" 🔹 Q: {question}")
        print(f" 🔸 A(기대정답): {expected_ans}")
        print(f" 🤖 A(LLM답변): {answer}")
        
        # 내가 쪼갠 데이터가 어떤 꼴로 들어갔는지 두 눈으로 확인하는 영역 (터미널 도배 방지를 위해 Top-3만, 150자로 요약 출력)
        print(" ▼ [검색된 실제 청크 데이터 확인 (Top-3)]")
        for idx, doc in enumerate(retrieved_docs[:3]):
            snippet = doc.page_content.replace('\n', ' ')[:150] + "..."
            print(f"   - 청크 {idx+1} [{doc.metadata.get('source_year')}년]: {snippet}")
            
        print(f" ▶ 검색된 청크 출처(Top-5): {retrieved_years}")
        print(f" ▶ 자동 판정: [청크 포함 여부: {chunk_inclusion}] | [최종 정답: {ans_mark}]")
        print("-" * 70)

    # ---------------- 요약 통계 ----------------
    print("\n" + "="*50)
    print("📈 [Basic RAG 평가 결과 요약]")
    print("="*50)
    print(f" - 총 문항 수: {total_q}")
    print(f" - 검색된 청크에 정답이 포함된 횟수: {chunks_included_count}/{total_q} ({(chunks_included_count/total_q)*100:.1f}%)")
    print(f" - LLM 최종 정답률: {correct_answers}/{total_q} ({(correct_answers/total_q)*100:.1f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    vs = indexing()
    rag_chain, rag_retriever = get_rag_chain(vs)
    evaluate_golden_dataset(rag_chain, rag_retriever)