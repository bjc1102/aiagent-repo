from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. PDF 로딩
loader = PyPDFLoader("../data/2024 알기 쉬운 의료급여제도.pdf")
docs = loader.load()

print(f"총 페이지 수: {len(docs)}")

# 2. 청킹
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

split_docs = text_splitter.split_documents(docs)

print(f"총 청크 수: {len(split_docs)}")

# 청크 샘플 출력 (과제용 핵심)
print("\n===== 청크 샘플 3개 =====")

for i in range(3):
    doc = split_docs[i]
    print(f"\n[Chunk {i+1}] page={doc.metadata.get('page')}")
    print(doc.page_content[:500].replace("\n", " "))

# 3. 임베딩 (OpenAI)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 4. 벡터 저장
vectorstore = FAISS.from_documents(split_docs, embedding_model)

vectorstore.save_local("faiss_index")

print("\nFAISS 저장 완료")