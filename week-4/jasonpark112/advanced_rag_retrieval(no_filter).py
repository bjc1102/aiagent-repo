import json
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GOLDEN_DATASET_PATH = os.path.join(BASE_DIR, "golden_dataset.jsonl")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
OUTPUT_FILE = os.path.join(BASE_DIR, "advanced_rag_no_filter_results.txt")

MD_FILES = {
    "2025": os.path.join(BASE_DIR, "../data/2025 알기 쉬운 의료급여제도.pdf_by_PaddleOCR-VL-1.5.md"),
    "2026": os.path.join(BASE_DIR, "../data/2026 알기 쉬운 의료급여제도.pdf_by_PaddleOCR-VL-1.5.md"),
}

# Hybrid Search 설정
VECTOR_K = 5
BM25_K = 5
VECTOR_WEIGHT = 0.5
BM25_WEIGHT = 0.5
RERANK_TOP_N = 3   # Re-ranking 후 LLM에 전달할 최종 청크 수

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""아래 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트에는 출처 년도가 표시되어 있습니다. 질문이 특정 년도를 묻는 경우 해당 년도의 정보만 사용하세요.
컨텍스트에 없는 내용은 "정보를 찾을 수 없습니다"라고 답하세요.
답변은 핵심 값만 간결하게 답하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
)


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
                all_child_docs.append(
                    Document(
                        page_content=child_text,
                        metadata={
                            "source_year": year,
                            "parent_id": f"{year}_p{p_idx}",
                            "parent_content": parent_content,
                            "child_index": c_idx,
                        }
                    )
                )
    print(f"BM25용 child 청크 로드 완료: {len(all_child_docs)}개")
    return all_child_docs


def build_context(docs):
    seen_parents = set()
    context_parts = []
    for doc in docs:
        parent_id = doc.metadata.get("parent_id", "")
        source_year = doc.metadata.get("source_year", "unknown")
        parent_content = doc.metadata.get("parent_content", doc.page_content)
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            context_parts.append(f"[출처 년도: {source_year}]\n{parent_content}")
    return "\n\n---\n\n".join(context_parts)


def rerank(question, docs, top_n):
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    pairs = [(question, doc.metadata.get("parent_content", doc.page_content)) for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def check_year_accuracy(docs, source_year):
    retrieved_years = [doc.metadata.get("source_year", "") for doc in docs]
    correct = any(y == source_year for y in retrieved_years)
    return correct, retrieved_years


def check_answer(llm_answer, expected_answer):
    return expected_answer.strip() in llm_answer.strip()


def load_golden_dataset(path):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def main():
    print("벡터스토어 로드 중...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_K})

    print("BM25 인덱스 구성 중...")
    child_docs = load_child_docs()
    bm25_retriever = BM25Retriever.from_documents(child_docs)
    bm25_retriever.k = BM25_K

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[VECTOR_WEIGHT, BM25_WEIGHT]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = RAG_PROMPT | llm

    golden_data = load_golden_dataset(GOLDEN_DATASET_PATH)
    total = len(golden_data)
    correct_count = 0
    year_correct_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for item in golden_data:
            qid = item["id"]
            question = item["question"]
            expected = item["expected_answer"]
            source_year = item.get("source_year", "")
            difficulty = item.get("difficulty", "")

            # Hybrid Search
            hybrid_docs = ensemble_retriever.invoke(question)

            # Re-ranking (년도 필터링 없음)
            reranked_docs = rerank(question, hybrid_docs, top_n=RERANK_TOP_N)

            # 년도 정확도 확인
            year_ok, retrieved_years = check_year_accuracy(reranked_docs, source_year)
            if source_year in ("2025+2026", "cross-year"):
                year_ok = True

            # 컨텍스트 구성 & LLM 생성
            context = build_context(reranked_docs)
            llm_response = chain.invoke({"context": context, "question": question})
            llm_answer = llm_response.content.strip()

            # 정답 판정
            is_correct = check_answer(llm_answer, expected)
            if is_correct:
                correct_count += 1
            if year_ok:
                year_correct_count += 1

            output = "\n========================\n"
            output += f"[{qid}] 난이도: {difficulty} | source_year: {source_year}\n"
            output += f"질문: {question}\n"
            output += f"정답: {expected}\n"
            output += f"LLM 답변: {llm_answer}\n"
            output += f"정답 여부: {'o' if is_correct else 'x'}\n"
            output += f"년도 검색 정확도: {'✓ 올바른 년도' if year_ok else f'✗ 년도 오류 (검색된 년도: {retrieved_years})'}\n"
            output += "\n[Re-ranking 후 최종 청크 (년도 필터링 없음)]\n"
            for i, doc in enumerate(reranked_docs, start=1):
                year = doc.metadata.get("source_year", "?")
                output += f"  Top{i} [year={year}] {doc.page_content[:100].replace(chr(10), ' ')}\n"

            print(output)
            out.write(output)

        summary = "\n========================\n"
        summary += f"[Advanced RAG (년도 필터링 없음) 결과 요약]\n"
        summary += f"BM25 k={BM25_K}, Vector k={VECTOR_K}, 가중치 vector:{VECTOR_WEIGHT} / BM25:{BM25_WEIGHT}\n"
        summary += f"Re-ranker: BAAI/bge-reranker-v2-m3, top_n={RERANK_TOP_N}\n"
        summary += f"년도 pre-filtering: 없음\n"
        summary += f"총 문항: {total}\n"
        summary += f"정답: {correct_count} / {total} ({correct_count / total:.1%})\n"
        summary += f"년도 검색 정확도: {year_correct_count} / {total} ({year_correct_count / total:.1%})\n"

        print(summary)
        out.write(summary)


if __name__ == "__main__":
    main()
