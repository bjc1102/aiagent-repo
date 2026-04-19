import json
import os
import time
from dotenv import load_dotenv

# 1. 기본 문서 처리 및 텍스트 스플리터
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2. 모델 및 DB 관련
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 3. 검색기(Retriever) 관련
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever

# 4. 리랭킹(Re-ranking) 관련
from langchain_cohere import CohereRerank
load_dotenv()

CHUNKING_SIZE = 500
CHUNKING_OVERLAP = 100
DB_PATH = "./chroma_db_week4"
GOLDEN_DATA_PATH = "golden_dataset.jsonl"

def indexing_and_load():
    print("🚀 [Step 2-1] Vector + BM25 하이브리드 인덱싱 (Re-ranking용 준비)...")
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

    if os.path.exists(DB_PATH):
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=DB_PATH)

    bm25_retriever = BM25Retriever.from_documents(all_splits)
    print("✅ 데이터 로드 완료!")
    return vectorstore, bm25_retriever

def get_answer_and_docs(question, vectorstore, bm25_retriever, llm, prompt):
    # 1. 메타데이터 필터 설정
    year_filter = None
    if "2025" in question and "2026" not in question: year_filter = "2025"
    elif "2026" in question and "2025" not in question: year_filter = "2026"

    # [핵심 변경점] 리랭킹을 위해 1차 검색(Hybrid)에서 k값을 10으로 늘려 넓게.
    search_kwargs = {"k": 10}
    if year_filter:
        search_kwargs["filter"] = {"source_year": year_filter}
        
    vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    bm25_retriever.k = 10

    # 2. 앙상블 리트리버 (Vector + BM25)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    # 3. [Step 2-3] Cohere Re-ranker 적용
    reranker = CohereRerank(
        model="rerank-v3.5", # 권장 모델
        top_n=3 # 넓게 가져온 10~20개 중 가장 관련성 높은 3개만 압축
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever
    )
    
    # 문서 검색 (이때 자동으로 Hybrid -> Rerank 압축 과정이 일어남)
    docs = compression_retriever.invoke(question)
    
    # LLM 컨텍스트 포맷팅 및 답변 생성
    formatted_context = "\n\n".join([f"[출처: {d.metadata['source_year']}년] {d.page_content}" for d in docs])
    chain = prompt | llm
    answer = chain.invoke({"context": formatted_context, "question": question}).content
    
    return answer, docs, year_filter

def evaluate_reranking_rag():
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

    print("\n📊 [Advanced RAG - Reranking] 골든 데이터셋 20문항 추출 시작...\n")
    
    with open(GOLDEN_DATA_PATH, 'r', encoding='utf-8') as f:
        golden_data = [json.loads(line) for line in f]

    total_q = len(golden_data)

    for i, item in enumerate(golden_data):
        q_id = f"q{i+1:02d}"
        question = item['question']
        expected_ans = item['expected_answer']
        target_year = item['source_year']
        difficulty = item['difficulty']

        answer, retrieved_docs, applied_filter = get_answer_and_docs(
            question, vectorstore, bm25_retriever, llm, prompt
        )
        
        # 리랭커를 통과하고 살아남은 최종 문서들의 출처와 점수
        retrieved_info = [f"{doc.metadata.get('source_year')} (점수: {doc.metadata.get('relevance_score', 0):.2f})" for doc in retrieved_docs]

        # ---------------- 결과 출력 (수동 채점용) ----------------
        print(f"[{q_id}] 난이도: {difficulty} | 목표 연도: {target_year}")
        print(f" 🔹 Q: {question}")
        print(f" 🔸 A(기대정답): {expected_ans}")
        print(f" 🤖 A(LLM답변): {answer}")
        print(f" ⚙️ [메타데이터 필터]: {applied_filter if applied_filter else '적용 안됨'}")
        
        print(" ▼ [리랭킹 후 최종 압축된 청크 데이터 확인 (Top-3)]")
        for idx, doc in enumerate(retrieved_docs):
            score = doc.metadata.get('relevance_score', 0)
            snippet = doc.page_content.replace('\n', ' ')[:150] + "..."
            print(f"   - {idx+1}위 [{doc.metadata.get('source_year')}년 / 관련도: {score:.2f}]: {snippet}")
            
        print(f" ▶ 최종 선별된 청크 출처(Top-K): {retrieved_info}")
        print("-" * 70)

        print("Cohere API 요금제 제한(1분 10회) 방지: 6.5초 대기 중...\n")
        time.sleep(6.5)

if __name__ == "__main__":
    evaluate_reranking_rag()