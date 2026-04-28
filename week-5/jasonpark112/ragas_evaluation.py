import json
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    ContextRecall,
    LLMContextPrecisionWithReference,
    Faithfulness,
    ResponseRelevancy,
    AnswerCorrectness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DATASET_PATH = os.path.join(BASE_DIR, "golden_dataset_v2.jsonl")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
MD_FILES = {
    "2025": os.path.join(BASE_DIR, "../data/2025 알기 쉬운 의료급여제도.pdf_by_PaddleOCR-VL-1.5.md"),
    "2026": os.path.join(BASE_DIR, "../data/2026 알기 쉬운 의료급여제도.pdf_by_PaddleOCR-VL-1.5.md"),
}

VECTOR_K = 5 
# 벡터 검색 시 상위 5개 청크 후보
BM25_K = 5
# BM25 검색 시 상위 5개 청크 후보
RERANK_TOP_N = 3
# Reranking 후 최종 3개만 컨텍스트로 사용

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 출처 년도가 표시되어 있습니다. 질문이 특정 년도를 묻는 경우 해당 년도의 정보만 사용하세요.
컨텍스트에 없는 내용은 "정보를 찾을 수 없습니다"라고 답하세요.
답변은 근거와 함께 충분히 설명하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
)


def load_golden_dataset(path):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

# 중복 제거 & 컨텍스트 구성
def build_context_and_contexts(docs):
    """parent_content 기준 중복 제거 후 context 문자열과 contexts 리스트 반환"""
    seen_parents = set()
    context_parts = []
    contexts_list = []
    for doc in docs:
        parent_id = doc.metadata.get("parent_id", "")
        source_year = doc.metadata.get("source_year", "unknown")
        parent_content = doc.metadata.get("parent_content", doc.page_content)
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            context_parts.append(f"[출처 년도: {source_year}]\n{parent_content}")
            contexts_list.append(parent_content)
    return "\n\n---\n\n".join(context_parts), contexts_list

# Parent-Child Chunking 전략을 사용
def load_child_docs():
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    all_child_docs = []
    for year, file_path in MD_FILES.items():
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        parent_chunks = parent_splitter.split_documents(docs)
        for p_idx, parent_doc in enumerate(parent_chunks):
            parent_content = parent_doc.page_content
            child_texts = child_splitter.split_text(parent_content)
            for c_idx, child_text in enumerate(child_texts):
                all_child_docs.append(Document(
                    page_content=child_text,
                    metadata={
                        "source_year": year,
                        "parent_id": f"{year}_p{p_idx}",
                        "parent_content": parent_content,
                        "child_index": c_idx,
                    }
                ))
    return all_child_docs

# Cross-Encoder 재정렬 (질문과 분서를 함께 입력해서 정밀한 관련도 점수를 계산 -> 앙상블 검색으로 얻은 후보들을 재정렬해 상위 Rerank top k 개만 선택)
# Bi-Encoder(FAISS)는 빠르지만 정밀도가 낮음
def rerank(question, docs, top_n):
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    pairs = [(question, doc.metadata.get("parent_content", doc.page_content)) for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]

# 단순 벡터 검색 (FAISS 코사인 유사도 검색만 사용하는 가장 기본적인 형태)
def run_basic_rag(question, vectorstore, llm_chain):
    retrieved_docs = vectorstore.similarity_search(question, k=3)
    context, retrieved_contexts = build_context_and_contexts(retrieved_docs)
    response = llm_chain.invoke({"context": context, "question": question}).content.strip()
    return response, retrieved_contexts

# 하이브리드 + Reranking 
def run_advanced_rag(question, ensemble_retriever, llm_chain):
    hybrid_docs = ensemble_retriever.invoke(question)
    reranked_docs = rerank(question, hybrid_docs, top_n=RERANK_TOP_N)
    context, retrieved_contexts = build_context_and_contexts(reranked_docs)
    response = llm_chain.invoke({"context": context, "question": question}).content.strip()
    return response, retrieved_contexts


def main():
    golden_data = load_golden_dataset(GOLDEN_DATASET_PATH)

    # RAG 파이프라인 세팅
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_chain = RAG_PROMPT | llm

    print("BM25 인덱스 구성 중...")
    child_docs = load_child_docs()
    bm25_retriever = BM25Retriever.from_documents(child_docs)
    bm25_retriever.k = BM25_K
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_K})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    # 평가용 LLM / 임베딩 세팅
    evaluator_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-5", temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    # Ragas 평가 지표 5가지
    metrics = [
        ContextRecall(), 
        LLMContextPrecisionWithReference(),
        Faithfulness(),
        ResponseRelevancy(),
        AnswerCorrectness(),
    ]

    # ragas 0.4.x에서 adapt_prompts 미지원 → Claude Sonnet이 한국어 텍스트를 직접 평가
    for metric in metrics:
        metric.llm = evaluator_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = evaluator_embeddings

    # Basic / Advanced RAG 실행 및 샘플 수집
    basic_samples = []
    advanced_samples = []

    for i, item in enumerate(golden_data):
        question = item["question"]
        reference = item["ground_truth"]
        reference_contexts = item["ground_truth_contexts"]

        print(f"[{i+1}/20] {question[:40]}...")

        # Basic / Advanced 각각 실행
        basic_response, basic_contexts = run_basic_rag(question, vectorstore, llm_chain)
        advanced_response, advanced_contexts = run_advanced_rag(question, ensemble_retriever, llm_chain)

        # SingleTurnSample : 질문 / 답변 / 검색컨텍스트 / 정답 묶음
        basic_samples.append(SingleTurnSample(
            user_input=question,
            response=basic_response,
            retrieved_contexts=basic_contexts,
            reference=reference,
            reference_contexts=reference_contexts,
        ))
        advanced_samples.append(SingleTurnSample(
            user_input=question,
            response=advanced_response,
            retrieved_contexts=advanced_contexts,
            reference=reference,
            reference_contexts=reference_contexts,
        ))

    basic_dataset = EvaluationDataset(samples=basic_samples)
    advanced_dataset = EvaluationDataset(samples=advanced_samples)

    # Ragas 평가 실행
    print("\nBasic RAG 평가 중...")
    basic_result = evaluate(dataset=basic_dataset, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings)
    basic_df = basic_result.to_pandas()
    basic_df.to_csv(os.path.join(BASE_DIR, "basic_ragas_scores.csv"), index=False, encoding="utf-8-sig")
    print("basic_ragas_scores.csv 저장 완료")

    print("\nAdvanced RAG 평가 중...")
    advanced_result = evaluate(dataset=advanced_dataset, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings)
    advanced_df = advanced_result.to_pandas()
    advanced_df.to_csv(os.path.join(BASE_DIR, "advanced_ragas_scores.csv"), index=False, encoding="utf-8-sig")
    print("advanced_ragas_scores.csv 저장 완료")

    print("\n=== Basic RAG 평균 ===")
    print(basic_df.mean(numeric_only=True))
    print("\n=== Advanced RAG 평균 ===")
    print(advanced_df.mean(numeric_only=True))


if __name__ == "__main__":
    main()
