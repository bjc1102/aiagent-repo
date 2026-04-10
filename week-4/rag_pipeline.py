from __future__ import annotations

import argparse
import getpass
import os
import re
from pathlib import Path
from typing import List, Optional

import cohere
import requests
from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "vectorstore" / "faiss_index"

PROVIDER_NAME = ""
MODEL_NAME = ""
API_KEY = ""
BASE_URL = ""
CLIENT: Optional[OpenAI] = None

COHERE_API_KEY = ""
COHERE_CLIENT = None


def ask_value(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def ask_secret(prompt: str) -> str:
    return getpass.getpass(f"{prompt}: ").strip()


def choose_provider() -> str:
    while True:
        print("사용할 LLM 제공자를 선택하세요.")
        print("1. OpenAI")
        print("2. Gemini")
        choice = input("번호 입력 [1/2]: ").strip()

        if choice == "1":
            return "openai"
        if choice == "2":
            return "gemini"

        print("잘못된 입력입니다. 1 또는 2를 입력하세요.\n")


def list_openai_models(api_key: str, base_url: str = "") -> List[str]:
    if base_url:
        temp_client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        temp_client = OpenAI(api_key=api_key)

    models = temp_client.models.list()
    model_ids = sorted([m.id for m in models.data])
    return sorted(set(model_ids))


def list_gemini_models(api_key: str) -> List[str]:
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    resp = requests.get(url, params={"key": api_key}, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    models = data.get("models", [])

    model_names: List[str] = []
    for model in models:
        name = model.get("name", "")
        if name.startswith("models/"):
            name = name.split("/", 1)[1]
        if name:
            model_names.append(name)

    return sorted(set(model_names))


def choose_model_from_list(model_names: List[str], title: str) -> str:
    if not model_names:
        raise ValueError(f"{title} 목록이 비어 있습니다.")

    print(f"\n=== {title} 목록 ===")
    for idx, name in enumerate(model_names, start=1):
        print(f"{idx}. {name}")

    while True:
        choice = input(f"{title} 모델 번호 선택 [1-{len(model_names)}]: ").strip()
        if not choice.isdigit():
            print("숫자를 입력하세요.")
            continue

        idx = int(choice)
        if 1 <= idx <= len(model_names):
            return model_names[idx - 1]

        print("범위를 벗어났습니다.")


def build_client() -> OpenAI:
    if not API_KEY:
        raise ValueError("API KEY 값이 없습니다.")

    if BASE_URL:
        return OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return OpenAI(api_key=API_KEY)


def build_cohere_client():
    global COHERE_API_KEY, COHERE_CLIENT

    if COHERE_CLIENT is not None:
        return COHERE_CLIENT

    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "").strip() or ask_secret("COHERE_API_KEY 입력")
    COHERE_CLIENT = cohere.ClientV2(api_key=COHERE_API_KEY)
    print("[INFO] Cohere client initialized.")
    return COHERE_CLIENT


def resolve_runtime_config(default_model: str = "") -> None:
    global PROVIDER_NAME, MODEL_NAME, API_KEY, BASE_URL, CLIENT

    PROVIDER_NAME = choose_provider()

    if PROVIDER_NAME == "openai":
        API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or ask_secret("OPENAI_API_KEY 입력")
        BASE_URL = ask_value("OPENAI_BASE_URL 입력 (없으면 엔터)", "")

        use_list = ask_value("OpenAI 모델 목록을 불러와서 선택할까요? (y/n)", "y").lower()
        if use_list == "y":
            try:
                models = list_openai_models(API_KEY, BASE_URL)
                MODEL_NAME = choose_model_from_list(models, "OpenAI")
            except Exception as e:
                print(f"[WARN] OpenAI 모델 목록 조회 실패: {e}")
                MODEL_NAME = ask_value("OpenAI MODEL_NAME 직접 입력", default_model or "gpt-4.1-mini")
        else:
            MODEL_NAME = ask_value("OpenAI MODEL_NAME 직접 입력", default_model or "gpt-4.1-mini")

    elif PROVIDER_NAME == "gemini":
        API_KEY = os.getenv("GEMINI_API_KEY", "").strip() or ask_secret("GEMINI_API_KEY 입력")
        BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

        use_list = ask_value("Gemini 모델 목록을 불러와서 선택할까요? (y/n)", "y").lower()
        if use_list == "y":
            try:
                models = list_gemini_models(API_KEY)
                MODEL_NAME = choose_model_from_list(models, "Gemini")
            except Exception as e:
                print(f"[WARN] Gemini 모델 목록 조회 실패: {e}")
                MODEL_NAME = ask_value(
                    "Gemini MODEL_NAME 직접 입력",
                    default_model or "gemini-3.1-flash-lite-preview",
                )
        else:
            MODEL_NAME = ask_value(
                "Gemini MODEL_NAME 직접 입력",
                default_model or "gemini-3.1-flash-lite-preview",
            )

    CLIENT = build_client()


def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def load_vectorstore(index_dir: Path) -> FAISS:
    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS 인덱스를 찾을 수 없습니다: {index_dir}\n"
            f"먼저 build_index.py를 실행하세요."
        )

    embeddings = build_embeddings()
    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def load_all_documents_from_faiss(vectorstore: FAISS) -> List[Document]:
    docstore = vectorstore.docstore._dict
    return list(docstore.values())


def build_bm25_retriever_from_vectorstore(vectorstore: FAISS, top_k: int = 4) -> BM25Retriever:
    docs = load_all_documents_from_faiss(vectorstore)
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = top_k
    return retriever


def extract_year_from_question(question: str) -> Optional[str]:
    match = re.search(r"(20\d{2})", question)
    return match.group(1) if match else None


def apply_year_filter(docs: List[Document], question: str) -> List[Document]:
    question_year = extract_year_from_question(question)
    if question_year is None:
        return docs

    filtered = [
        doc for doc in docs
        if str(doc.metadata.get("source_year")) == question_year
    ]
    return filtered if filtered else docs


def deduplicate_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    unique_docs = []

    for doc in docs:
        key = (
            str(doc.metadata.get("source_file")),
            str(doc.metadata.get("page")),
            doc.page_content[:150],
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


def search_documents(
    vectorstore: FAISS,
    question: str,
    top_k: int = 4,
    use_year_filter: bool = False,
) -> List[Document]:
    retrieved_docs = vectorstore.similarity_search(question, k=top_k * 2)

    if use_year_filter:
        retrieved_docs = apply_year_filter(retrieved_docs, question)

    return retrieved_docs[:top_k]


def search_documents_hybrid(
    vectorstore: FAISS,
    question: str,
    top_k: int = 4,
    use_year_filter: bool = False,
    vector_k: int = 6,
    bm25_k: int = 6,
    debug: bool = False,
) -> List[Document]:
    vector_docs = vectorstore.similarity_search(question, k=vector_k)

    bm25_retriever = build_bm25_retriever_from_vectorstore(vectorstore, top_k=bm25_k)
    bm25_docs = bm25_retriever.invoke(question)

    if debug:
        print("\n=== Vector Docs ===")
        for i, doc in enumerate(vector_docs, 1):
            print(
                i,
                f"year={doc.metadata.get('source_year')}",
                f"page={doc.metadata.get('page')}",
                doc.page_content[:120].replace("\n", " ")
            )

        print("\n=== BM25 Docs ===")
        for i, doc in enumerate(bm25_docs, 1):
            print(
                i,
                f"year={doc.metadata.get('source_year')}",
                f"page={doc.metadata.get('page')}",
                doc.page_content[:120].replace("\n", " ")
            )

    if use_year_filter:
        vector_docs = apply_year_filter(vector_docs, question)
        bm25_docs = apply_year_filter(bm25_docs, question)

    # BM25 우선 병합
    merged_docs = deduplicate_docs(bm25_docs + vector_docs)

    return merged_docs[:top_k]


def rerank_documents_with_cohere(
    question: str,
    docs: List[Document],
    top_k: int = 4,
    model: str = "rerank-v3.5",
) -> List[Document]:
    if not docs:
        return []

    client = build_cohere_client()
    texts = [doc.page_content for doc in docs]

    print("\n=== Cohere Rerank Request ===")
    print(f"model={model}")
    print(f"num_candidates={len(texts)}")
    print(f"top_n={min(top_k, len(texts))}")

    try:
        response = client.rerank(
            model=model,
            query=question,
            documents=texts,
            top_n=min(top_k, len(texts)),
        )

        reranked_docs: List[Document] = []
        for item in response.results:
            reranked_docs.append(docs[item.index])

        return reranked_docs

    except Exception as e:
        print(f"[WARN] Cohere rerank failed: {e}")
        return docs[:top_k]


def search_documents_hybrid_rerank(
    vectorstore: FAISS,
    question: str,
    top_k: int = 4,
    use_year_filter: bool = False,
    vector_k: int = 6,
    bm25_k: int = 6,
    rerank_model: str = "rerank-v3.5",
    debug: bool = False,
) -> List[Document]:
    hybrid_docs = search_documents_hybrid(
        vectorstore=vectorstore,
        question=question,
        top_k=max(top_k * 2, 8),
        use_year_filter=use_year_filter,
        vector_k=vector_k,
        bm25_k=bm25_k,
        debug=debug,
    )

    reranked_docs = rerank_documents_with_cohere(
        question=question,
        docs=hybrid_docs,
        top_k=top_k,
        model=rerank_model,
    )

    if debug:
        print("\n=== Cohere Reranked Docs ===")
        for i, doc in enumerate(reranked_docs, 1):
            print(
                i,
                f"year={doc.metadata.get('source_year')}",
                f"page={doc.metadata.get('page')}",
                doc.page_content[:120].replace("\n", " ")
            )

    return reranked_docs


def format_context(docs: List[Document]) -> str:
    if not docs:
        return "검색된 문서가 없습니다."

    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        source_year = doc.metadata.get("source_year", "unknown")
        source_file = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        content = doc.page_content.strip()

        block = (
            f"[문서 {i}]\n"
            f"- source_year: {source_year}\n"
            f"- source_file: {source_file}\n"
            f"- page: {page}\n"
            f"- content:\n{content}"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_messages(question: str, context: str):
    system_prompt = (
        "당신은 의료급여제도 문서를 기반으로 답변하는 RAG 도우미입니다.\n"
        "반드시 검색 문맥만 근거로 답변하세요.\n"
        "문맥에 없는 내용을 추측하지 마세요.\n"
        "질문에 특정 연도(예: 2025년, 2026년)가 포함되어 있으면 반드시 해당 연도의 문맥을 우선적으로 사용하세요.\n"
        "검색 문맥에 서로 다른 연도의 정보가 섞여 있으면 질문의 연도와 일치하는 정보만 사용하세요.\n"
        "답변 마지막에는 사용한 근거의 source_year와 page를 간단히 정리하세요."
    )

    user_prompt = (
        f"[질문]\n{question}\n\n"
        f"[검색 문맥]\n{context}\n\n"
        "[답변 형식]\n"
        "1. 질문에 대한 간결한 답변\n"
        "2. 필요한 경우 한두 문장 설명\n"
        '3. 마지막 줄에 "근거: 2026년 p.12, 2026년 p.13" 형태로 표시'
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(question: str, docs: List[Document]) -> str:
    if CLIENT is None:
        return "[LLM generation skipped] client가 초기화되지 않았습니다."

    context = format_context(docs)
    messages = build_messages(question, context)

    try:
        resp = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            messages=messages,
            max_tokens=800,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return (
            "[LLM generation failed]\n"
            f"provider={PROVIDER_NAME}\n"
            f"model={MODEL_NAME}\n"
            f"error={e}\n\n"
            "[Retrieved context]\n"
            f"{context}"
        )


def print_retrieved_docs(docs: List[Document]) -> None:
    print("\n=== Retrieved Documents ===")
    if not docs:
        print("검색 결과 없음")
        return

    for i, doc in enumerate(docs, start=1):
        preview = doc.page_content[:200].replace("\n", " ")
        print(
            f"[{i}] "
            f"year={doc.metadata.get('source_year')} | "
            f"file={doc.metadata.get('source_file')} | "
            f"page={doc.metadata.get('page')} | "
            f"preview={preview}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        type=str,
        default="2026년 의료급여에서 장기지속형 주사제의 본인부담률은 얼마인가?",
        help="질문을 입력하세요.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="검색할 청크 수",
    )
    parser.add_argument(
        "--use_year_filter",
        action="store_true",
        help="질문에 연도가 있으면 해당 연도 청크를 우선 필터링",
    )
    parser.add_argument(
        "--default_model",
        type=str,
        default="",
        help="목록 조회 실패 시 기본 모델명",
    )
    parser.add_argument(
        "--search_mode",
        type=str,
        default="vector",
        choices=["vector", "hybrid", "hybrid_rerank"],
        help="검색 방식 선택",
    )
    parser.add_argument(
        "--debug_search",
        action="store_true",
        help="vector/bm25/rerank 중간 결과를 출력",
    )
    args = parser.parse_args()

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"INDEX_DIR: {INDEX_DIR}")
    print(f"QUESTION: {args.question}")
    print(f"TOP_K: {args.top_k}")
    print(f"USE_YEAR_FILTER: {args.use_year_filter}")
    print(f"SEARCH_MODE: {args.search_mode}")

    resolve_runtime_config(default_model=args.default_model)

    print("\n=== Runtime Config ===")
    print(f"PROVIDER: {PROVIDER_NAME}")
    print(f"MODEL   : {MODEL_NAME}")
    print(f"BASE_URL: {BASE_URL or '(default)'}")

    vectorstore = load_vectorstore(INDEX_DIR)

    if args.search_mode == "vector":
        docs = search_documents(
            vectorstore=vectorstore,
            question=args.question,
            top_k=args.top_k,
            use_year_filter=args.use_year_filter,
        )
    elif args.search_mode == "hybrid":
        docs = search_documents_hybrid(
            vectorstore=vectorstore,
            question=args.question,
            top_k=args.top_k,
            use_year_filter=args.use_year_filter,
            debug=args.debug_search,
        )
    else:
        docs = search_documents_hybrid_rerank(
            vectorstore=vectorstore,
            question=args.question,
            top_k=args.top_k,
            use_year_filter=args.use_year_filter,
            debug=args.debug_search,
        )

    print_retrieved_docs(docs)

    answer = generate_answer(
        question=args.question,
        docs=docs,
    )

    print("\n=== Final Answer ===")
    print(answer)


if __name__ == "__main__":
    main()