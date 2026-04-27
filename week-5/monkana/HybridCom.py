import json
from pathlib import Path
from typing import Any, Callable, Optional, Union

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

import BasicRag
import HybridRerank


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result" / "HybridTest"
DATASET_PATH = BASE_DIR / "golden_dataset_step2_pilot_5.jsonl"

VECTOR_K = HybridRerank.VECTOR_K
BM25_K = HybridRerank.BM25_K
COMPRESSION_TOP = 8


def build_contextual_compression_retriever(base_retriever):
    compressor = LLMChainExtractor.from_llm(HybridRerank.get_llm())
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def retrieve_docs_hybrid_compression(
    vectorstore,
    all_documents: list[Document],
    question: str,
    source_year: str = "",
    top_n: int = COMPRESSION_TOP,
) -> list[Document]:
    source_year = source_year or HybridRerank.infer_source_year(question)

    if "," in source_year:
        years = [y.strip() for y in source_year.split(",") if y.strip()]
        docs: list[Document] = []
        per_year_top = max(1, top_n // len(years))
        for yr in years:
            ensemble = HybridRerank.build_hybrid_rerank_retriever(
                vectorstore,
                all_documents,
                yr,
                vector_k=VECTOR_K,
                bm25_k=BM25_K,
            )
            compression_retriever = build_contextual_compression_retriever(ensemble)
            docs.extend(compression_retriever.invoke(question)[:per_year_top])
        return docs

    ensemble = HybridRerank.build_hybrid_rerank_retriever(
        vectorstore,
        all_documents,
        source_year,
        vector_k=VECTOR_K,
        bm25_k=BM25_K,
    )
    compression_retriever = build_contextual_compression_retriever(ensemble)
    return compression_retriever.invoke(question)[:top_n]


def ask_question_hybrid_compression(
    vectorstore,
    all_documents: list[Document],
    question: str,
    source_year: str = "",
    top_n: int = COMPRESSION_TOP,
    retrieved_docs: Optional[list[Document]] = None,
) -> str:
    if retrieved_docs is None:
        retrieved_docs = retrieve_docs_hybrid_compression(
            vectorstore,
            all_documents,
            question,
            source_year=source_year,
            top_n=top_n,
        )

    _, context = HybridRerank.format_retrieved_contexts(retrieved_docs)
    prompt = HybridRerank.build_prompt(question, context)
    return str(HybridRerank.get_llm().invoke(prompt).content)


def run_pipeline(
    vectorstore,
    all_documents: list[Document],
    question: str,
    source_year: str = "",
    top_n: int = COMPRESSION_TOP,
) -> dict[str, Any]:
    retrieved_docs = retrieve_docs_hybrid_compression(
        vectorstore,
        all_documents,
        question,
        source_year=source_year,
        top_n=top_n,
    )
    retrieved_contexts, _ = HybridRerank.format_retrieved_contexts(retrieved_docs)
    response = ask_question_hybrid_compression(
        vectorstore,
        all_documents,
        question,
        source_year=source_year,
        top_n=top_n,
        retrieved_docs=retrieved_docs,
    )

    return {
        "question": question,
        "source_year": source_year or HybridRerank.infer_source_year(question),
        "retrieved_docs": retrieved_docs,
        "retrieved_contexts": retrieved_contexts,
        "response": response,
    }


def evaluate_golden_dataset(
    vectorstore,
    all_documents: list[Document],
    jsonl_path: Union[str, Path],
    *,
    search_label: str = "Hybrid (Vector+BM25) + Contextual Compression",
    output_filename: str = "evaluation_hybrid_com.txt",
    pipeline_name: str = "Hybrid+Compression",
    retrieve_fn: Optional[Callable[..., list[Document]]] = None,
    answer_fn: Optional[Callable[..., str]] = None,
    top_n: int = COMPRESSION_TOP,
    extra_settings: Optional[list[tuple[str, str]]] = None,
):
    retrieve_fn = retrieve_fn or retrieve_docs_hybrid_compression
    answer_fn = answer_fn or ask_question_hybrid_compression

    rows = []
    detail_lines = []

    with open(jsonl_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(items, start=1):
        q_id = item["id"]
        question = item["question"]
        expected = item["ground_truth"]
        difficulty = item["difficulty"]
        source_year = item.get("source_year", "")
        print(f"[{pipeline_name}] {index}/{len(items)} 처리 중: {q_id}")

        retrieved_docs = retrieve_fn(
            vectorstore,
            all_documents,
            question,
            source_year=source_year,
            top_n=top_n,
        )
        _, context_text = HybridRerank.format_retrieved_contexts(retrieved_docs)
        search_ok = bool(retrieved_docs)
        year_ok = HybridRerank.check_year_correctness(retrieved_docs, source_year)

        answer = answer_fn(
            vectorstore,
            all_documents,
            question,
            source_year=source_year,
            top_n=top_n,
            retrieved_docs=retrieved_docs,
        )
        answer_ok_flag, judge_result = HybridRerank.judge_cross_year_answer(
            question,
            expected,
            answer,
            context_text,
        )
        verdict = "정답" if answer_ok_flag else "오답"

        rows.append(
            {
                "q_id": q_id,
                "difficulty": difficulty,
                "source_year": source_year,
                "search": "O" if search_ok else "X",
                "year": "O" if year_ok else "X",
                "answer": answer.replace("\n", " ")[:80],
                "expected": expected.replace("\n", " ")[:80],
                "verdict": verdict,
                "judge": judge_result.replace("\n", " ")[:120],
            }
        )

        retrieved_years_str = ", ".join(
            doc.metadata.get("source_year", "?") for doc in retrieved_docs
        )
        detail_lines.append("=" * 70)
        detail_lines.append(
            f"질문 ID   : {q_id}  |  난이도: {difficulty}  |  source_year: {source_year}"
        )
        detail_lines.append(f"질문      : {question}")
        detail_lines.append(f"검색 방식 : {search_label}")
        detail_lines.append(f"검색 청크 출처년도: [{retrieved_years_str}]")
        detail_lines.append("검색된 청크:")
        for idx, doc in enumerate(retrieved_docs, start=1):
            yr = doc.metadata.get("source_year", "?")
            detail_lines.append(
                f"\n  [청크 {idx}] (출처년도={yr}, page={doc.metadata.get('page')})"
            )
            detail_lines.append("  " + doc.page_content.replace("\n", "\n  "))
        detail_lines.append(f"검색 성공 : {'O' if search_ok else 'X'}")
        detail_lines.append(f"년도 정확 : {'O' if year_ok else 'X'}")
        detail_lines.append(f"예상 답변 : {expected}")
        detail_lines.append(f"모델 답변 : {answer}")
        if judge_result:
            detail_lines.append(f"LLM 채점  : {judge_result}")
        detail_lines.append(f"판정      : {verdict}")
        detail_lines.append("")
        print(f"[{pipeline_name}] {q_id} 완료: {verdict}")

    summary_lines = []
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append(f"## {pipeline_name} 기록 테이블\n")

    summary_lines.append("### 설정값\n")
    summary_lines.append("| 항목 | 설정값 |")
    summary_lines.append("|------|--------|")
    summary_lines.append(f"| BM25 Retriever k | {BM25_K} |")
    summary_lines.append(f"| Vector Retriever k | {VECTOR_K} |")
    summary_lines.append(
        f"| Ensemble 가중치 (vector : BM25) | {HybridRerank.VECTOR_WEIGHT} : {HybridRerank.BM25_WEIGHT} |"
    )
    summary_lines.append(
        f"| Compression 방식 | LLMChainExtractor / {HybridRerank.CHAT_MODEL} |"
    )
    summary_lines.append(f"| Compression 후 최종 Top-K | {top_n} |")
    summary_lines.append(
        "| 메타데이터 필터링 | 단일 년도 질문 시 source_year 필터 적용 |"
    )
    if extra_settings:
        for key, value in extra_settings:
            summary_lines.append(f"| {key} | {value} |")

    summary_lines.append("\n### 문항별 결과\n")
    summary_lines.append(
        "| 질문 ID | 난이도 | source_year | 검색 방식 | 검색 결과 존재 여부 | 올바른 년도 검색 여부 | LLM 생성 답변 | 정답 | 정답 여부 |"
    )
    summary_lines.append(
        "|---------|--------|-------------|----------|-------------------|-------------------|-------------|------|----------|"
    )
    for r in rows:
        summary_lines.append(
            f"| {HybridRerank.table_cell(r['q_id'])} | {HybridRerank.table_cell(r['difficulty'])} | {HybridRerank.table_cell(r['source_year'])} "
            f"| {pipeline_name} | {r['search']} | {r['year']} "
            f"| {HybridRerank.table_cell(r['answer'], 60)} | {HybridRerank.table_cell(r['expected'], 60)} "
            f"| {HybridRerank.table_cell(r['verdict'])} |"
        )

    total = len(rows)
    correct = sum(1 for r in rows if r["verdict"] == "정답")
    year_ok_count = sum(1 for r in rows if r["year"] == "O")
    search_ok_count = sum(1 for r in rows if r["search"] == "O")

    summary_lines.append(
        f"\n| **{pipeline_name} 정답률** | | | | | | | | **{correct}/{total}** |"
    )

    summary_lines.append("\n### 난이도별 정답률\n")
    summary_lines.append("| 난이도 | 정답 | 전체 | 정답률 |")
    summary_lines.append("|--------|------|------|--------|")
    for diff in ["easy", "medium", "hard", "cross-year"]:
        diff_rows = [r for r in rows if r["difficulty"] == diff]
        if diff_rows:
            d_correct = sum(1 for r in diff_rows if r["verdict"] == "정답")
            summary_lines.append(
                f"| {diff} | {d_correct} | {len(diff_rows)} | {d_correct/len(diff_rows)*100:.0f}% |"
            )

    summary_lines.append("\n### 검색/년도 진단\n")
    summary_lines.append("| 항목 | 값 |")
    summary_lines.append("|------|-----|")
    summary_lines.append(f"| 검색 성공률 | {search_ok_count}/{total} |")
    summary_lines.append(f"| 올바른 년도 검색 성공률 | {year_ok_count}/{total} |")

    all_output = detail_lines + summary_lines
    output_path = RESULT_DIR / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_output))

    print(f"저장 완료: {output_path}")
    if total == 0:
        print("평가할 항목이 없습니다.")
        return
    print(f"\n[{pipeline_name} 결과 요약]")
    print(f"  정답률      : {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  년도 정확도  : {year_ok_count}/{total} ({year_ok_count/total*100:.1f}%)")


def main():
    print("ChromaDB 로드 중...")
    vectorstore = BasicRag.load_vectorstore()

    print("BM25 인덱스용 문서 로드 중...")
    all_documents = HybridRerank.load_all_documents()
    print(f"  총 {len(all_documents)}개 청크 로드 완료")

    evaluate_golden_dataset(
        vectorstore,
        all_documents,
        DATASET_PATH,
    )


if __name__ == "__main__":
    main()
