import json
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.core.retriever import get_relevant_documents

def check_retrieval_quality():
    jsonl_path = base_dir / "golden_dataset.jsonl"
    
    if not jsonl_path.exists():
        print(f"❌ 오류: Golden Dataset 파일을 찾을 수 없습니다: {jsonl_path}")
        return

    print("=" * 70)
    print("🎯 Step 3: 검색 품질 확인 (Top-5 Retrieval Check)")
    print("=" * 70)

    results = []
    success_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            q_id = data["id"]
            question = data["question"]
            evidence = data["evidence_text"]
            difficulty = data["difficulty"]

            print(f"\n🔍 [{q_id}] ({difficulty}) {question}")
            
            # 1. 검색 수행 (Top-5)
            # 이전에 만든 크로마 DB에서 검색합니다.
            docs = get_relevant_documents(question, k=5)
            
            # 2. 결과 확인 (evidence_text 포함 여부)
            # 현실적으로 '정확한 포함' 보다는 근거가 충분히 들어있는지 키워드 등으로 볼 수도 있지만,
            # 여기서는 텍스트 내 존재 여부로 판단합니다.
            full_context = "\n".join([doc.page_content for doc in docs])
            
            # 부분 문자열 일치 검사 (단순화를 위해)
            # evidence_text의 핵심 키워드가 포함되어 있는지 확인
            is_success = False
            # evidence_text가 실제 원문에서 복사된 것이라면 바로 일치할 것이고, 
            # 요약된 것이라면 핵심 단어 위주로 체크해야 합니다.
            # 여기서는 사용자 제공 evidence_text가 직접적으로 내용에 포함되는지 봅니다.
            if evidence in full_context:
                is_success = True
            else:
                # 간단한 키워드 체크 (예외 처리: evidence_text가 길어서 부분적으로만 맞을 때)
                keywords = [k.strip() for k in evidence.split("→") if k.strip()]
                if not keywords: keywords = [evidence]
                if all(kw in full_context for kw in keywords):
                    is_success = True

            status = "🚀 성공" if is_success else "❌ 실패"
            if is_success:
                success_count += 1
            
            # 검색된 청크 요약 (앞부분 100자)
            chunk_summary = docs[0].page_content[:100].replace("\n", " ") + "..." if docs else "N/A"
            
            results.append({
                "id": q_id,
                "difficulty": difficulty,
                "status": status,
                "summary": chunk_summary
            })
            
            print(f"결과: {status}")
            if not is_success:
                print(f"💡 기대 근거: {evidence}")

    # 결과 리포트 출력
    print("\n" + "=" * 70)
    print("📊 종합 결과")
    print("-" * 70)
    print(f"| 질문 ID | 난이도 | 검색 결과 | 검색된 청크 요약 |")
    print("|---------|--------|----------|----------------|")
    for res in results:
        print(f"| {res['id']} | {res['difficulty']} | {res['status']} | {res['summary']} |")
    print("-" * 70)
    print(f"✅ 검색 성공률: {success_count}/{len(results)} ({(success_count/len(results))*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    check_retrieval_quality()
