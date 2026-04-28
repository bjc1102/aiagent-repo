import json
import os
from langchain_community.document_loaders import TextLoader
# md 파일을 텍스트로 읽어오는 로더
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 문서를 청크로 나누는 스플리터
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# 텍스트 -> 벡터로 변환, GPT 모델 호출
from langchain_community.vectorstores import FAISS
# 벡터 유사도 검색 인덱스
from langchain_community.retrievers import BM25Retriever
# 키워드 기반 검색
from langchain_classic.retrievers import EnsembleRetriever
# hybrid search + rrf 내장
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
# Re-ranking 용 모델

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 현재 이 파일이 있는 폴더 경로
GOLDEN_DATASET_PATH = os.path.join(BASE_DIR, "golden_dataset.jsonl")
# 평가용 질문 + 정답 데이터 (.jsonl 형식)
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
# 미리 만들어준 벡터 인덱스 저장 위치
OUTPUT_FILE = os.path.join(BASE_DIR, "advanced_rag_yes_filter_results.txt")
# 평가 결과를 저장할 텍스트 파일
MD_FILES = {
    "2025": os.path.join(BASE_DIR, "../data/2025 알기 쉬운 의료급여제도.pdf_by_PaddleOCR-VL-1.5.md"),
    "2026": os.path.join(BASE_DIR, "../data/2026 알기 쉬운 의료급여제도.pdf_by_PaddleOCR-VL-1.5.md"),
}

# Hybrid Search 설정
VECTOR_K = 5
# 벡터 검색 상위 몇 개
BM25_K = 5
# BM25 검색 상위 몇 개
VECTOR_WEIGHT = 0.5
# 벡터 검색 RRF 가중치
BM25_WEIGHT = 0.5
# BM25 검색 RRF 가중치
RERANK_TOP_N = 3   
# Re-ranking 후 LLM에 전달할 최종 청크 수

#  정리하자면 벡터, BM25 각각 5개씩 검색 -> RRF로 합산 -> CrossEncoder로 재정렬 후 최종 3개만 LLM에 전달

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


# -------------------------------
# 문서 로드 및 Parent-Child 청킹 (BM25 인덱스 재구성용)
# -------------------------------
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
# 2025, 2026 문서 각각 루프
# 각 Parent Child로 나눌 때 parent_content를 메타데이터에 저장 
# 나중에 검색은 Child로 하지만, LLM엔 Parent를 넘기기 위해서

def build_context(docs):
    """parent_content 기반 컨텍스트 구성 (중복 parent 제거)"""
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
# Re-ranking Child 문서들을 받아서 seen_parents set으로 중복 Parent 제거
# LLM에는 Child가 아닌 Parent 전체 내용을 컨텍스트로 전달
# 각 컨텍스트 앞에 출처 년도 태그 붙임



def rerank(question, docs, top_n):
    """CrossEncoder(bge-reranker-v2-m3)로 re-ranking (parent_content 기준)"""
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    pairs = [(question, doc.metadata.get("parent_content", doc.page_content)) for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]
# CrossEncoder -> 질문과 문서를 쌍으로 넣어서 관련도 점수 계산
# pairs -> [(질문, 문서1내용), (질문, 문서2내용)]
# scores -> 각 쌍의 관련도 점수 리스트
# 점수 높은 순으로 정렬 후 상위 3개만 반환


def filter_by_year(docs, source_year):
    """cross-year가 아닌 경우 해당 년도 문서만 필터링, 결과 없으면 원본 반환"""
    if source_year in ("2025+2026", "cross-year", ""):
        return docs
    filtered = [d for d in docs if d.metadata.get("source_year", "") == source_year]
    return filtered if filtered else docs
# 메타데이터 필터


def check_year_accuracy(docs, source_year):
    retrieved_years = [doc.metadata.get("source_year", "") for doc in docs]
    correct = any(y == source_year for y in retrieved_years)
    return correct, retrieved_years
# Re-ranking 된 문서들 중 올바른 년도가 있는지 확인


def check_answer(llm_answer, expected_answer):
    return expected_answer.strip() in llm_answer.strip()
# LLM 답변에 정답 문자열이 포함되는지 확인 (단순 포함 여부)


def load_golden_dataset(path):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def main():
    # 1. 임베딩 & 벡터스토어 로드
    print("벡터스토어 로드 중...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_K})
    # 미리 만들어둔 FAISS 인덱스를 불러옴. 매번 새로 임베딩 안 해도 됨

    # 2. BM25 인덱스 구성 (메모리 기반, 매 실행 시 재구성)
    print("BM25 인덱스 구성 중...")
    child_docs = load_child_docs()
    bm25_retriever = BM25Retriever.from_documents(child_docs)
    bm25_retriever.k = BM25_K
    # 매 실행마다 child_docs로 BM25 인덱스를 메모리에서 새로 구성 (저장 안함) 

    # 3. EnsembleRetriever (Hybrid Search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[VECTOR_WEIGHT, BM25_WEIGHT]
    )
    # 두 retriever를 묶어서 hybrid + rrf 완성

    # 4. LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = RAG_PROMPT | llm
    # temperature=0 -> 창의성 없이 일관된 답변 생성 | 연산자로 프롬프트 -> LLM 체인 연결

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

            # 5. Hybrid Search
            hybrid_docs = ensemble_retriever.invoke(question)
            # 여기서 실질적으로 rff 계산

            # 5-1. 년도 pre-filtering (메타데이터 필터)
            hybrid_docs = filter_by_year(hybrid_docs, source_year)

            # 6. Re-ranking
            reranked_docs = rerank(question, hybrid_docs, top_n=RERANK_TOP_N)

            # 7. 년도 정확도 확인
            year_ok, retrieved_years = check_year_accuracy(reranked_docs, source_year)
            if source_year in ("2025+2026", "cross-year"):
                year_ok = True

            # 8. 컨텍스트 구성 & LLM 생성
            context = build_context(reranked_docs)
            llm_response = chain.invoke({"context": context, "question": question})
            llm_answer = llm_response.content.strip()

            # 9. 정답 판정
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
            output += "\n[Re-ranking 후 최종 청크]\n"
            for i, doc in enumerate(reranked_docs, start=1):
                year = doc.metadata.get("source_year", "?")
                output += f"  Top{i} [year={year}] {doc.page_content[:100].replace(chr(10), ' ')}\n"

            print(output)
            out.write(output)

        summary = "\n========================\n"
        summary += f"[Advanced RAG 결과 요약]\n"
        summary += f"BM25 k={BM25_K}, Vector k={VECTOR_K}, 가중치 vector:{VECTOR_WEIGHT} / BM25:{BM25_WEIGHT}\n"
        summary += f"Re-ranker: BAAI/bge-reranker-v2-m3, top_n={RERANK_TOP_N}\n"
        summary += f"총 문항: {total}\n"
        summary += f"정답: {correct_count} / {total} ({correct_count / total:.1%})\n"
        summary += f"년도 검색 정확도: {year_correct_count} / {total} ({year_correct_count / total:.1%})\n"

        print(summary)
        out.write(summary)


if __name__ == "__main__":
    main()
