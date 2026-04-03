import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    load_dotenv()

    dataset_path = Path("golden_dataset.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(f"golden dataset 파일이 없습니다: {dataset_path}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    golden_dataset = load_jsonl(str(dataset_path))

    success_count = 0
    #Top-K 
    k = 5

    print(f"[INFO] total questions: {len(golden_dataset)}")
    print(f"[INFO] top-k: {k}")

    for row in golden_dataset:
        qid = row["id"]
        question = row["question"]
        evidence_text = row["evidence_text"]

        docs = vectorstore.similarity_search(question, k=k)
        joined_text = "\n".join(doc.page_content for doc in docs)

        success = evidence_text.replace(" ", "") in joined_text.replace(" ", "")
        if success:
            success_count += 1

        print("\n" + "=" * 100)
        print(f"id: {qid}")
        print(f"difficulty: {row['difficulty']}")
        print(f"question: {question}")
        print(f"expected evidence: {evidence_text}")
        print(f"retrieval result: {'성공' if success else '실패'}")

        for i, doc in enumerate(docs, start=1):
            print("\n" + "-" * 80)
            print(f"[TOP {i}] page={doc.metadata.get('page')}")
            print(doc.page_content[:500])

    print("\n" + "=" * 100)
    print(f"검색 성공률: {success_count}/{len(golden_dataset)}")


if __name__ == "__main__":
    main()