from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MD_FILES = {
    "2025": os.path.join(BASE_DIR, "../data/2025 알기 쉬운 의료급여제도.pdf_by_PaddleOCR-VL-1.5.md"),
    "2026": os.path.join(BASE_DIR, "../data/2026 알기 쉬운 의료급여제도.pdf_by_PaddleOCR-VL-1.5.md"),
}

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=150
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=80
)

all_child_docs = []

for year, file_path in MD_FILES.items():
    print(f"\n[{year}] 로딩: {file_path}")
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    parent_chunks = parent_splitter.split_documents(docs)
    print(f"  Parent 청크 수: {len(parent_chunks)}")

    child_count = 0
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
            child_count += 1

    print(f"  Child 청크 수: {child_count}")

print(f"\n전체 Child 청크 수: {len(all_child_docs)}")

print("\n===== 샘플 (Child 청크) =====")
for i in range(min(5, len(all_child_docs))):
    d = all_child_docs[i]
    print(f"\n[{i+1}] year={d.metadata['source_year']} | parent_id={d.metadata['parent_id']}")
    print(f"  child: {d.page_content[:200].replace(chr(10), ' ')}")
    print(f"  parent(앞100자): {d.metadata['parent_content'][:100].replace(chr(10), ' ')}")
    print("-" * 60)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.from_documents(all_child_docs, embedding_model)
vectorstore.save_local("faiss_index_c1000_p3000")

print("\nFAISS 저장 완료 → faiss_index_c1000_p3000")
