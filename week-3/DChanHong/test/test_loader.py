# os, sys: 파이썬의 운영체제 및 시스템 환경을 제어하기 위해 쓰이는 내장 모듈입니다. (아래에서 sys.path 경로 추가에 사용)
import os
import sys
# Path: 파일의 경로를 직접 조작하기 위해 사용하는 내장 모듈입니다.
from pathlib import Path

# 파이프라인에서 src 패키지를 인식할 수 있도록 경로 추가 (test 폴더 안이므로 .parent.parent 사용)
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

# 이전에 작성한 PDF 로드용 함수를 가져옵니다.
from src.core.loader import load_pdf

def test_pdf_parsing():
    # PDF 파일 경로 (절대 경로)
    pdf_path = "/Users/root1/Desktop/aiagent-repo/week-3/data/2024 알기 쉬운 의료급여제도.pdf"
    
    print(f"테스트 대상: {pdf_path}")
    print("-" * 50)
    
    try:
        # 이전에 작성한 core.loader 모듈을 활용하여 파싱
        documents = load_pdf(pdf_path)
        print(f"✅ 로드 성공! 총 {len(documents)} 페이지가 파싱되었습니다.")
        
        # 첫 2 페이지 내용 확인
        for i in range(min(2, len(documents))):
            print(f"\n[페이지 {i+1} 샘플]")
            # 내용이 너무 길어질 수 있으므로 200자까지만 출력
            preview = documents[i].page_content[:200]
            print(preview + ("..." if len(preview) == 200 else ""))
            
        print("\n[메타데이터 샘플 (페이지 1)]")
        print(documents[0].metadata)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    test_pdf_parsing()
