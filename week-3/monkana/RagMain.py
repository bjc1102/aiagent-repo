import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils import CHROMA_DIR, reset_chroma_dir
from rank_bm25 import BM25Okapi


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

REBUILD_DB = True


def get_embeddings():
    return OpenAIEmbeddings(model=EMBED_MODEL)


def get_llm():
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)

REBUILD_DB = True
def build_vectorstore():
    reset_chroma_dir()

    with open(BASE_DIR / "all_chunks.json", encoding="utf-8") as f:
        all_chunks = json.load(f)

    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name="medical_pdf",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    documents = []
    ids = []

    for i, chunk in enumerate(all_chunks):
        tables_text = ""
        if chunk["tables"]:
            tables_text = " ".join(chunk["tables"])[:600]

        # content + text + tables 모두 embed
        page_content = f"{chunk['content']}\n{chunk['text']}\n{tables_text}".strip()

        doc = Document(
            page_content=page_content,
            metadata={
                "parent_section": chunk["parent_section"],
                "section": chunk["section"],
                "page": chunk["page"],
                "text": chunk["text"],
                "tables": json.dumps(chunk["tables"], ensure_ascii=False),
            }
        )
        documents.append(doc)
        ids.append(f"chunk::{i}")

    vectorstore.add_documents(documents=documents, ids=ids)
    print(f"vector db build complete: {len(documents)} chunks")
    return vectorstore


def load_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        collection_name="medical_pdf",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

def tokenize(text: str) -> list[str]:
    tokens = text.split()
    korean = re.findall(r'[가-힣]{2,}', text)
    english = re.findall(r'[A-Za-z]+', text)
    return tokens + korean + english



def ask_question(vectorstore, question: str, k: int = 10):
    candidate_docs = vectorstore.similarity_search(question, k=k)

    # BM25 재랭킹
    corpus = [tokenize(doc.page_content) for doc in candidate_docs]
    bm25 = BM25Okapi(corpus)
    query_tokens = tokenize(question)
    bm25_scores = bm25.get_scores(query_tokens)

    scored_docs = sorted(
        [(bm25_scores[i], doc) for i, doc in enumerate(candidate_docs)],
        key=lambda x: x[0], reverse=True
    )
    retrieved_docs = [doc for _, doc in scored_docs[:4]]

    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        tables = json.loads(doc.metadata.get("tables", "[]"))
        table_str = "\n".join(tables)
        full_context = f"{doc.page_content}\n{doc.metadata.get('text', '')}\n{table_str}".strip()
        context_parts.append(f"[문서 {i}] page={doc.metadata.get('page')}\n{full_context}")

    context = "\n\n".join(context_parts)

    prompt = f"""
너는 의료급여제도 PDF를 기반으로 답변하는 도우미다.
반드시 아래 검색 문맥만 바탕으로 답변해라.
문맥에 없으면 없다고 말해라.
단답으로 대답할 것. 예시 답변: 45,000원, 해당 되지 않음, 15%, 무료 등

[질문]
{question}

[검색 문맥]
{context}
""".strip()

    llm = get_llm()
    response = llm.invoke(prompt)
    # RagMain.py ask_question 함수에서 디버깅 추가
    return response.content


def evaluate_golden_dataset(vectorstore, jsonl_path: str):
    success_count = 0
    total_count = 0
    output_lines = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            question = item["question"]
            expected = item["expected_answer"]
            evidence_text = item["evidence_text"]
            difficulty = item["difficulty"]
            q_id = item["id"]

            evidence_tokens = [t.strip().replace(" ", "") for t in evidence_text.split(",")]

            # 1. 검색
            candidate_docs = vectorstore.similarity_search(question, k=10)

            corpus = [tokenize(doc.page_content) for doc in candidate_docs]
            bm25 = BM25Okapi(corpus)
            query_tokens = tokenize(question)
            bm25_scores = bm25.get_scores(query_tokens)

            scored_docs = sorted(
                [(bm25_scores[i], doc) for i, doc in enumerate(candidate_docs)],
                key=lambda x: x[0], reverse=True
            )
            retrieved_docs = [doc for _, doc in scored_docs[:4]]

            # 2. full_retrieved_text 구성
            full_retrieved_text = ""
            content_list = []
            for doc in retrieved_docs:
                full_retrieved_text += doc.page_content
                full_retrieved_text += doc.metadata.get("text", "")
                full_retrieved_text += doc.metadata.get("tables", "")
                content_list.append(doc.page_content)

            # 3. missing_tokens 체크
            missing_tokens = [
                t for t in evidence_tokens
                if t.replace(" ", "") not in full_retrieved_text.replace(" ", "")
            ]
            search_success = "O" if not missing_tokens else "X"
            if not missing_tokens:
                success_count += 1
            total_count += 1

            answer = ask_question(vectorstore, question, k=10)
            answer_match = "O" if expected.strip().replace(" ", "") in answer.strip().replace(" ", "") else "X"

            output_lines.append(f"{'='*60}")
            output_lines.append(f"질문 ID  : {q_id}")
            output_lines.append(f"난이도   : {difficulty}")
            output_lines.append(f"질문     : {question}")
            output_lines.append(f"검색된 청크:")
            for idx, content in enumerate(content_list, start=1):
                output_lines.append(f"\n[검색 청크 {idx}]")
                output_lines.append(content)
                output_lines.append("")
            output_lines.append(f"누락 토큰: {', '.join(missing_tokens) if missing_tokens else '-'}")
            output_lines.append(f"검색 결과: {search_success}")
            output_lines.append(f"예상 답변: {expected}")
            output_lines.append(f"모델 답변: {answer}")
            output_lines.append(f"답변 일치: {answer_match}")
            output_lines.append("")

    output_lines.append(f"{'='*60}")
    output_lines.append(f"검색 성공률: {success_count}/{total_count}")

    output_path = BASE_DIR / "evaluation_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"저장 완료: {output_path}")





def main():
    if REBUILD_DB:
        vectorstore = build_vectorstore()
    else:
        vectorstore = load_vectorstore()

    evaluate_golden_dataset(vectorstore, BASE_DIR / "goldenDataset.jsonl")



if __name__ == "__main__":
    main()

