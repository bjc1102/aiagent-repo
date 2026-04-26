import json
import os
import time
import pandas as pd
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
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

# --- 설정 ---
DB_PATH = "./chroma_db_week5"
GOLDEN_DATA_PATH = "golden_dataset_v2.jsonl"
PILOT_MODE = False



def indexing_and_load():
    print("[인덱싱] 문서 로드 및 DB/BM25 구축 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    all_splits = []
    for year in ["2025", "2026"]:
        loader = PDFPlumberLoader(f"data/{year} 알기 쉬운 의료급여제도.pdf")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_year"] = year
        splits = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        ).split_documents(docs)
        all_splits.extend(splits)

    if os.path.exists(DB_PATH):
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(
            documents=all_splits, embedding=embeddings, persist_directory=DB_PATH
        )

    bm25_retriever = BM25Retriever.from_documents(all_splits)
    return vectorstore, bm25_retriever


def run_basic_rag(question, target_year, vectorstore, llm, prompt):
    docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(question)
    context = "\n\n".join(
        [f"[출처: {d.metadata['source_year']}년] {d.page_content}" for d in docs]
    )
    answer = (prompt | llm).invoke({"context": context, "question": question}).content
    return answer, [d.page_content for d in docs]


def run_advanced_rag(question, target_year, vectorstore, bm25_retriever, llm, prompt):
    year_filter = None
    if "2025" in question and "2026" not in question:
        year_filter = "2025"
    elif "2026" in question and "2025" not in question:
        year_filter = "2026"

    search_kwargs = {"k": 10}
    if year_filter:
        search_kwargs["filter"] = {"source_year": year_filter}
    vector_docs = vectorstore.as_retriever(search_kwargs=search_kwargs).invoke(question)

    bm25_retriever.k = 10
    raw_bm25_docs = bm25_retriever.invoke(question)
    bm25_docs = (
        [d for d in raw_bm25_docs if d.metadata.get("source_year") == year_filter]
        if year_filter
        else raw_bm25_docs
    )

    unique_docs = list({d.page_content: d for d in (vector_docs + bm25_docs)}.values())

    if unique_docs:
        reranker = CohereRerank(model="rerank-v3.5", top_n=3)
        final_docs = reranker.compress_documents(documents=unique_docs, query=question)
    else:
        final_docs = []

    context = "\n\n".join(
        [f"[출처: {d.metadata['source_year']}년] {d.page_content}" for d in final_docs]
    )
    answer = (prompt | llm).invoke({"context": context, "question": question}).content
    return answer, [d.page_content for d in final_docs]


def evaluate_pipelines():
    vectorstore, bm25_retriever = indexing_and_load()

    generator_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 출처 년도(source_year)가 포함되어 있습니다. 질문이 묻는 년도와 일치하는 정보를 우선적으로 사용하세요.
정보가 부족하면 "정보를 찾을 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
    )

    # 평가용 LLM: gpt-4o-mini (생성용과 다른 모델 패밀리, 높은 rate limit, 저렴)
    eval_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0)
    )
    eval_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    golden_data = []
    with open(GOLDEN_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            golden_data.append(json.loads(line))

    if PILOT_MODE:
        golden_data = golden_data[:3]

    basic_samples, advanced_samples = [], []

    print(f"\n[Step 1] Basic / Advanced RAG 답변 추출 시작 (총 {len(golden_data)}문항)")
    for i, item in enumerate(golden_data):
        q = item["question"]
        gt = item["ground_truth"]
        gt_contexts = item["ground_truth_contexts"]
        target_year = item["source_year"]

        print(f"  ({i+1}/{len(golden_data)}) 질문 처리 중: {q[:30]}...")

        b_ans, b_ctx = run_basic_rag(q, target_year, vectorstore, generator_llm, prompt)
        basic_samples.append(
            SingleTurnSample(
                user_input=q, response=b_ans, retrieved_contexts=b_ctx,
                reference=gt, reference_contexts=gt_contexts,
            )
        )

        a_ans, a_ctx = run_advanced_rag(
            q, target_year, vectorstore, bm25_retriever, generator_llm, prompt
        )
        advanced_samples.append(
            SingleTurnSample(
                user_input=q, response=a_ans, retrieved_contexts=a_ctx,
                reference=gt, reference_contexts=gt_contexts,
            )
        )

        time.sleep(3)

    metrics = [
        ContextRecall(),
        LLMContextPrecisionWithReference(),
        Faithfulness(),
        ResponseRelevancy(),
        AnswerCorrectness(),
    ]

    # gpt-4o-mini: rate limit 여유로워서 max_workers=4로 병렬 실행 가능
    run_cfg = RunConfig(max_workers=4, timeout=60, max_retries=10, max_wait=30)

    print("\n[Step 2] GPT-4o-mini가 Basic RAG를 채점 중입니다...")
    ds_basic = EvaluationDataset(samples=basic_samples)
    result_basic = evaluate(
        dataset=ds_basic, metrics=metrics, llm=eval_llm,
        embeddings=eval_embeddings, run_config=run_cfg,
    )
    result_basic.to_pandas().to_csv("basic_ragas_scores.csv", index=False, encoding="utf-8-sig")
    print("  => basic_ragas_scores.csv 저장 완료!")

    print("\n[Step 2] GPT-4o-mini가 Advanced RAG를 채점 중입니다...")
    ds_advanced = EvaluationDataset(samples=advanced_samples)
    result_advanced = evaluate(
        dataset=ds_advanced, metrics=metrics, llm=eval_llm,
        embeddings=eval_embeddings, run_config=run_cfg,
    )
    result_advanced.to_pandas().to_csv("advanced_ragas_scores.csv", index=False, encoding="utf-8-sig")
    print("  => advanced_ragas_scores.csv 저장 완료")

    print("\n모든 자동 평가가 완료되었습니다")


if __name__ == "__main__":
    evaluate_pipelines()
