import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

GOLDEN_DATASET_PATH = "golden_dataset.jsonl"
FAISS_INDEX_PATH = "faiss_index"
OUTPUT_FILE = "basic_rag_results.txt"
TOP_K = 3

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


def load_golden_dataset(path):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def build_context(retrieved_docs):
    """child 청크에서 parent_content를 꺼내 source_year와 함께 컨텍스트 구성 (중복 parent 제거)"""
    seen_parents = set()
    context_parts = []

    for doc in retrieved_docs:
        parent_id = doc.metadata.get("parent_id", "")
        source_year = doc.metadata.get("source_year", "unknown")
        parent_content = doc.metadata.get("parent_content", doc.page_content)

        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            context_parts.append(f"[출처 년도: {source_year}]\n{parent_content}")

    return "\n\n---\n\n".join(context_parts)


def check_year_accuracy(retrieved_docs, source_year):
    """검색된 청크 중 올바른 source_year가 하나라도 있는지 확인"""
    retrieved_years = [doc.metadata.get("source_year", "") for doc in retrieved_docs]
    correct = any(y == source_year for y in retrieved_years)
    return correct, retrieved_years


def check_answer(llm_answer, expected_answer):
    """LLM 답변에 expected_answer 핵심값이 포함되는지 확인"""
    return expected_answer.strip() in llm_answer.strip()


def main():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
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

            # 1. Top-K 검색
            retrieved_docs = vectorstore.similarity_search(question, k=TOP_K)

            # 2. 년도 정확도 확인
            year_ok, retrieved_years = check_year_accuracy(retrieved_docs, source_year)
            if source_year in ("2025+2026", "cross-year"):
                year_ok = True  # cross-year는 판정 제외

            # 3. 컨텍스트 구성 (parent_content + source_year)
            context = build_context(retrieved_docs)

            # 4. LLM 생성
            llm_response = chain.invoke({"context": context, "question": question})
            llm_answer = llm_response.content.strip()

            # 5. 정답 판정
            is_correct = check_answer(llm_answer, expected)
            if is_correct:
                correct_count += 1
            if year_ok:
                year_correct_count += 1

            # 출력 구성
            output = "\n========================\n"
            output += f"[{qid}] 난이도: {difficulty} | source_year: {source_year}\n"
            output += f"질문: {question}\n"
            output += f"정답: {expected}\n"
            output += f"LLM 답변: {llm_answer}\n"
            output += f"정답 여부: {'✓ 정답' if is_correct else '✗ 오답'}\n"
            output += f"년도 검색 정확도: {'✓ 올바른 년도' if year_ok else f'✗ 년도 오류 (검색된 년도: {retrieved_years})'}\n"
            output += "\n[검색된 청크]\n"
            for i, doc in enumerate(retrieved_docs, start=1):
                year = doc.metadata.get("source_year", "?")
                output += f"  Top{i} [year={year}] {doc.page_content[:100].replace(chr(10), ' ')}\n"

            print(output)
            out.write(output)

        # 최종 요약
        summary = "\n========================\n"
        summary += f"[Basic RAG 결과 요약]\n"
        summary += f"총 문항: {total}\n"
        summary += f"정답: {correct_count} / {total} ({correct_count / total:.1%})\n"
        summary += f"년도 검색 정확도: {year_correct_count} / {total} ({year_correct_count / total:.1%})\n"

        print(summary)
        out.write(summary)


if __name__ == "__main__":
    main()
