import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

GOLDEN_DATASET_PATH = "golden_dataset.jsonl"
FAISS_INDEX_PATH = "faiss_index"
OUTPUT_FILE = "retrieval_results_pdf.txt"
TOP_K = 3


def load_golden_dataset(path):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def normalize_text(text):
    return " ".join(text.split())


def main():
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    golden_data = load_golden_dataset(GOLDEN_DATASET_PATH)

    success_count = 0

    # txt 파일 열기
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for item in golden_data:
            question = item["question"]
            evidence_text = item["evidence_text"]

            retrieved_docs = vectorstore.similarity_search(question, k=TOP_K)

            found = False
            matched_chunk = None

            norm_evidence = normalize_text(evidence_text)

            # 출력 문자열 만들기
            output = "\n========================\n"
            output += f"질문: {question}\n"
            output += f"정답 근거(evidence): {evidence_text}\n"

            for i, doc in enumerate(retrieved_docs, start=1):
                chunk_text = doc.page_content
                page = doc.metadata.get("page", None)

                output += f"\n[Top {i}] page={page}\n"
                output += chunk_text[:2000].replace("\n", " ") + "\n"

                if norm_evidence in normalize_text(chunk_text):
                    found = True
                    matched_chunk = chunk_text

            output += "\n[결과]\n"

            if found:
                output += "검색 성공\n"
                output += "\n[매칭된 근거 청크]\n"
                output += matched_chunk[:1000].replace("\n", " ") + "\n"
                success_count += 1
            else:
                output += "검색 실패\n"

            # 콘솔 출력
            print(output)

            # 파일 저장
            out.write(output)

        total = len(golden_data)
        summary = "\n========================\n"
        summary += f"총 {total}문항\n"
        summary += f"성공 {success_count}\n"
        summary += f"성공률 {success_count / total:.2%}\n"

        print(summary)
        out.write(summary)


if __name__ == "__main__":
    main()