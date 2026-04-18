import json
import os
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.services.rag_service import ask_question

def test_rag_with_golden_dataset():
    jsonl_path = base_dir / "golden_dataset.jsonl"
    result_dir = base_dir / "result"
    
    # 결과 저장용 폴더 생성
    result_dir.mkdir(parents=True, exist_ok=True)
    
    if not jsonl_path.exists():
        print(f"❌ 오류: Golden Dataset 파일을 찾을 수 없습니다: {jsonl_path}")
        return

    print("=" * 80)
    print("🚀 Gemini (gemini-3-flash-preview) RAG 전수 테스트 및 결과 저장")
    print("=" * 80)

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            q_id = data["id"]
            question = data["question"]
            golden_answer = data["expected_answer"]

            print(f"\n🔍 [{q_id}] 질문: {question}")
            print(f"🎯 모범 답안: {golden_answer}")
            print("-" * 40)
            
            result_item = {
                "id": q_id,
                "question": question,
                "golden_dataset": {
                    "expected_answer": golden_answer,
                    "source_section": data.get("source_section"),
                    "evidence_text": data.get("evidence_text"),
                    "conditions": data.get("conditions")
                },
                "ai_response": None,
                "error": None
            }

            try:
                response = ask_question(question)
                print(f"✅ AI 답변: {response.expected_answer}")
                print(f"📍 근거 섹션: {response.source_section}")
                
                # AI 답변 결과를 객체에 저장
                result_item["ai_response"] = response.model_dump()
                
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                result_item["error"] = str(e)
            
            all_results.append(result_item)
            print("-" * 80)

    # 최종 결과를 JSON 파일로 저장
    result_file_path = result_dir / f"{timestamp}.json"
    with open(result_file_path, "w", encoding="utf-8") as rf:
        json.dump(all_results, rf, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 테스트 완료 및 결과 저장 성공!")
    print(f"📂 저장 경로: {result_file_path}")

if __name__ == "__main__":
    test_rag_with_golden_dataset()
