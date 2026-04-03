# os, sys: 터미널 환경이나 시스템 변수(sys.path 등)에 접근할 때 사용하는 내장 모듈입니다.
import os
import sys
# Path: 객체지향적인 파일 경로 조작을 다루는 내장 라이브러리입니다.
from pathlib import Path

# 파이프라인에서 src 패키지를 인식할 수 있도록 경로 추가 (test 폴더 안이므로 .parent.parent 사용)
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

# 프로젝트 내의 데이터 로더 및 분할 관련 함수들을 가져옵니다.
from src.core.loader import load_pdf
from src.core.splitter import split_documents

def test_pdf_chunking():
    pdf_path = "/Users/root1/Desktop/aiagent-repo/week-3/data/2024 알기 쉬운 의료급여제도.pdf"
    
    print("=" * 60)
    print("🚀 청킹(Chunking) 테스트 시작")
    print("=" * 60)

    try:
        # 1. 문서 로드
        documents = load_pdf(pdf_path)
        print(f"✅ 문서 로드 성공! (총 페이지 수: {len(documents)})")
        
        # 2. 문서 청킹 (설정된 chunk_size, chunk_overlap 사용)
        print("\n[문서 자르기(Chunk) 진행 중...]")
        chunks = split_documents(documents)
        
        print("\n" + "=" * 60)
        print(f"📊 청킹 결과 요약")
        print("=" * 60)
        print(f"총 청크(Chunk) 개수 : {len(chunks)} 개")
        print("-" * 60)

        # 3. 청킹 결과 일부 확인 (특히 중간 페이지 등 표가 있을 법한 곳 샘플링)
        # 5번째부터 3개 정도의 청크를 샘플로 출력해봅니다.
        start_idx = 5
        end_idx = min(start_idx + 3, len(chunks))
        
        for i in range(start_idx, end_idx):
            print(f"\n🏷️ [청크 번호 {i+1}] - 출처: {chunks[i].metadata.get('page_label')} 페이지")
            preview = chunks[i].page_content[:300] # 내용을 300자까지 확인
            print(f"내용:\n{preview} ...")
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    test_pdf_chunking()
