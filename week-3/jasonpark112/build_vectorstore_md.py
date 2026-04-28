from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import re

# -------------------------------
# 1. Markdown 로딩
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "../data/2024 알기 쉬운 의료급여제도.md")

loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

print(f"로드된 문서 수: {len(docs)}")

# -------------------------------
# 2. 구분선(---, ----, ----- 등) 판별
# -------------------------------
DELIM_RE = re.compile(r"^\s*-{3,}\s*$")

def is_delimiter_line(line: str) -> bool:
    return bool(DELIM_RE.match(line))

# -------------------------------
# 3. 구분선 기준 청킹
# ---
# 내용 A
# ---
# 내용 B
# ---
# =>
# chunk1 = 내용 A
# chunk2 = 내용 B
# -------------------------------
def split_boundary_sections(text: str):
    lines = text.splitlines()
    parts = []

    buffer = []

    for line in lines:
        if is_delimiter_line(line):
            # 구분선 만나면 지금까지 쌓인 내용 하나의 section으로 저장
            content = "\n".join(buffer).strip()
            if content:
                parts.append(("section", content))
            buffer = []
        else:
            buffer.append(line)

    # 마지막 남은 내용 처리
    content = "\n".join(buffer).strip()
    if content:
        parts.append(("section", content))

    return parts

# -------------------------------
# 4. 일반 텍스트 splitter
# 필요 시 fallback용
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

split_docs = []

for doc in docs:
    parts = split_boundary_sections(doc.page_content)

    print("\n===== PART DEBUG =====")
    for idx, (ptype, content) in enumerate(parts[:10], 1):
        print(f"[PART {idx}] type={ptype}")
        print(content[:300].replace("\n", " "))
        print("=" * 80)

    for part_type, content in parts:
        if part_type == "section":
            # --- 경계 사이의 내용을 통째로 1 chunk
            split_docs.append(
                Document(
                    page_content=content,
                    metadata={**doc.metadata, "type": "section"}
                )
            )
        else:
            # 현재 로직상 거의 안 쓰이지만 fallback
            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                split_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={**doc.metadata, "type": "text"}
                    )
                )

print(f"총 청크 수: {len(split_docs)}")

# -------------------------------
# 5. 샘플 출력
# -------------------------------
print("\n===== 샘플 =====")
for i in range(min(10, len(split_docs))):
    d = split_docs[i]
    print(f"\n[{i+1}] type={d.metadata['type']}")
    print(d.page_content[:5000].replace("\n", " "))
    print("-" * 60)

# -------------------------------
# 6. 임베딩 & 저장
# -------------------------------
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = FAISS.from_documents(split_docs, embedding_model)
vectorstore.save_local("faiss_index")

print("\nFAISS 저장 완료")