from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def main() -> None:
    load_dotenv()

    pdf_path = Path("data/2024 알기 쉬운 의료급여제도.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    # 1) PDF 로딩
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"[INFO] loaded pages: {len(documents)}")

    # 2) 청킹
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    print(f"[INFO] total chunks: {len(chunks)}")

    # 샘플 확인
    for i, chunk in enumerate(chunks[:3], start=1):
        print("\n" + "=" * 80)
        print(f"[CHUNK {i}] page={chunk.metadata.get('page')}")
        print(chunk.page_content[:700])

    # 3) 임베딩
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4) 벡터 저장소 생성
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5) 로컬 저장
    save_dir = "faiss_index"
    vectorstore.save_local(save_dir)
    print(f"\n[INFO] FAISS index saved to: {save_dir}")

    # 6) 검색 테스트
    query = "65세 이상 1종 수급권자가 틀니를 하면 본인부담률은 몇 퍼센트인가요?"
    results = vectorstore.similarity_search(query, k=3)

    print("\n" + "=" * 80)
    print(f"[QUERY] {query}")
    for i, doc in enumerate(results, start=1):
        print("\n" + "-" * 80)
        print(f"[TOP {i}] page={doc.metadata.get('page')}")
        print(doc.page_content[:700])


if __name__ == "__main__":
    main()