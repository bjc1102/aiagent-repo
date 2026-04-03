# os, sys: 시스템 경로에 프로젝트 루트를 추가하기 위한 내장 모듈
import os
import sys
# Path: 객체지향적인 파일 경로 조작을 다루는 내장 라이브러리
from pathlib import Path

# 파이프라인에서 src 패키지를 인식할 수 있도록 경로 추가 (현재 파일이 test/ 폴더 안에 있으므로 .parent.parent 사용)
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

# 파이프라인 실행 함수 및 DB 확인용 함수 가져오기
from src.pipeline import run_indexing_pipeline
from src.core.embedder import load_vector_store

def test_full_pipeline():
    # PDF 파일 경로 (현재 User 경로와 데이터 폴더 구조에 맞춤)
    # 만약 경로가 다르다면 본인 환경에 맞게 수정해주세요.
    pdf_path = "/Users/hong/Desktop/aiagent-repo/week-3/data/2024 알기 쉬운 의료급여제도.pdf"
    
    # 1. 대상 파일이 실제로 존재하는지 확인
    if not Path(pdf_path).exists():
        print(f"❌ 오류: PDF 파일을 찾을 수 없습니다. 경로를 확인해주세요.\n[입력된 경로]: {pdf_path}")
        return

    print("=" * 60)
    print("🚀 전체 파이프라인 (PDF 로드 ➔ 청킹 ➔ 임베딩 ➔ ChromaDB 저장) 테스트 시작")
    print("=" * 60)

    try:
        # 2. 전체 파이프라인 작동
        # 파이프라인 모듈 뒤에서 자동으로 문서 로드, 분할, 임베딩 후 persist_directory(storage)에 저장까지 수행합니다.
        run_indexing_pipeline(source=pdf_path, is_directory=False)
        print("\n✅ 파이프라인 실행 자체는 에러 없이 완료되었습니다!")
        
        # 3. 데이터가 DB에 실제로 잘 들어갔는지 간단히 확인해보기
        print("\n" + "=" * 60)
        print("💾 Chroma DB 저장 결과 확인")
        print("=" * 60)
        
        db = load_vector_store()
        data = db.get()
        ids = data.get('ids', [])
        
        if len(ids) > 0:
            print(f"🎉 성공! 총 {len(ids)}개의 임베딩 데이터가 Chroma DB에 무사히 저장되었습니다.")
        else:
            print("⚠️ 파이프라인은 완료되었으나, DB에 저장된 데이터 갯수가 0개입니다. (로직 확인 필요)")

    except Exception as e:
        print(f"\n❌ 파이프라인 실행 중 오류 발생:\n{e}")

if __name__ == "__main__":
    test_full_pipeline()
