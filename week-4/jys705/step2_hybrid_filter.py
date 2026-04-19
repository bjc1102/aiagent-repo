import json
import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
# --- Advanced RAG 용 추가 임포트 --- 
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

CHUNKING_SIZE = 500
CHUNKING_OVERLAP = 100
DB_PATH = "./chroma_db_week4"
GOLDEN_DATA_PATH = "golden_dataset.jsonl"

def indexing_and_load():
    """
    [Step 2-1] ChromaDB(벡터)와 BM25(키워드)를 동시에 준비.
    BM25는 DB 저장이 안 되므로 매번 원문을 로드해서 메모리에 올려야.
    """
    print("🚀 [Step 2-1 & 2-2] Vector + BM25 하이브리드 인덱싱 시작...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    years = ["2025", "2026"]
    all_splits = []
    
    # 1. 문서 로드 및 청킹
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

    # 2. Vector DB (Chroma) 구성
    if os.path.exists(DB_PATH):
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=DB_PATH)

    # 3. BM25 리트리버 구성 (메모리)
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 5
    print("✅ 벡터 DB 및 BM25(메모리) 구성 완료")
    
    return vectorstore, bm25_retriever

def get_answer_and_docs(question, vectorstore, bm25_retriever, llm, prompt):
    """
    질문마다 '연도'를 파악하여 동적으로 필터를 걸고 하이브리드 검색을 수행.
    """
    # [Step 2-2] 질문에서 연도(2025, 2026) 동적 추출 및 필터링 적용 
    year_filter = None
    if "2025" in question and "2026" not in question:
        year_filter = "2025"
    elif "2026" in question and "2025" not in question:
        year_filter = "2026"

    search_kwargs = {"k": 5}
    if year_filter:
        search_kwargs["filter"] = {"source_year": year_filter}
        
    # 벡터 리트리버에 필터 적용
    vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # [Step 2-1] 하이브리드 (Ensemble) 병합 (5:5 비율) 
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    # 문서 검색
    docs = ensemble_retriever.invoke(question)
    
    # LLM 컨텍스트 포맷팅 및 답변 생성
    formatted_context = "\n\n".join([f"[출처: {d.metadata['source_year']}년] {d.page_content}" for d in docs])
    chain = prompt | llm
    answer = chain.invoke({"context": formatted_context, "question": question}).content
    
    return answer, docs, year_filter

def evaluate_advanced_rag():
    vectorstore, bm25_retriever = indexing_and_load()
    
    template = """아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 출처 년도(source_year)가 포함되어 있습니다. 질문이 묻는 년도와 일치하는 정보를 우선적으로 사용하세요.
정보가 부족하면 "정보를 찾을 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    print("\n📊 [Advanced RAG] 골든 데이터셋 20문항 추출 시작...\n")
    
    with open(GOLDEN_DATA_PATH, 'r', encoding='utf-8') as f:
        golden_data = [json.loads(line) for line in f]

    total_q = len(golden_data)

    for i, item in enumerate(golden_data):
        q_id = f"q{i+1:02d}"
        question = item['question']
        expected_ans = item['expected_answer']
        target_year = item['source_year']
        difficulty = item['difficulty']

        # 하이브리드 + 필터링 검색 및 LLM 답변 도출
        answer, retrieved_docs, applied_filter = get_answer_and_docs(
            question, vectorstore, bm25_retriever, llm, prompt
        )
        
        retrieved_years = [doc.metadata.get('source_year', 'Unknown') for doc in retrieved_docs]

        # ---------------- 결과 출력 (수동 채점용) ----------------
        print(f"[{q_id}] 난이도: {difficulty} | 목표 연도: {target_year}")
        print(f" 🔹 Q: {question}")
        print(f" 🔸 A(기대정답): {expected_ans}")
        print(f" 🤖 A(LLM답변): {answer}")
        print(f" ⚙️ [적용된 메타데이터 필터]: {applied_filter if applied_filter else '적용 안됨 (교차비교 등)'}")
        
        print(" ▼ [검색된 실제 청크 데이터 확인 (Top-3)]")
        for idx, doc in enumerate(retrieved_docs[:3]):
            snippet = doc.page_content.replace('\n', ' ')[:150] + "..."
            print(f"   - 청크 {idx+1} [{doc.metadata.get('source_year')}년]: {snippet}")
            
        print(f" ▶ 검색된 청크 출처(Top-K): {retrieved_years}")
        print("-" * 70)

    print("\n" + "="*50)
    print("📈 [Advanced RAG (Hybrid + Filter) 실행 완료]")
    print("="*50)
    print(f" - 총 {total_q} 문항의 응답 및 검색 결과가 성공적으로 추출되었습니다.")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_advanced_rag()