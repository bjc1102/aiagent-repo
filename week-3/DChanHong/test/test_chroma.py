# os, sys: 시스템 경로에 프로젝트 루트를 추가하기 위한 모듈
import os
import sys
# Path: 객체지향적인 파일 경로 조작을 다루는 내장 라이브러리
from pathlib import Path

# 파이프라인에서 src 패키지를 인식할 수 있도록 경로 추가 (현재 파일이 test/ 폴더 안에 있으므로 parent.parent 사용)
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

# 크로마 DB 로드 함수 가져오기
from src.core.embedder import load_vector_store

def view_chroma_data():
    try:
        # 저장된 DB를 불러옵니다.
        db = load_vector_store()
        
        # DB 안에 들어있는 전체 데이터를 가져옵니다.
        data = db.get()
        
        # 데이터는 Dictionary 형태로, 각각의 리스트를 반환합니다.
        ids = data.get('ids', [])
        documents = data.get('documents', [])
        metadatas = data.get('metadatas', [])
        
        total_count = len(ids)
        print(f"\n📊 크로마 DB 조회 결과: 총 {total_count}개의 데이터(청크)가 저장되어 있습니다.\n")
        
        if total_count == 0:
            print("저장된 데이터가 없습니다. 먼저 인덱싱(main.py)을 실행해주세요!")
            return

        # 처음 2개의 데이터만 샘플로 출력해보기
        for i in range(min(2, total_count)):
            print("-" * 50)
            print(f"🔸 [데이터 ID]: {ids[i]}")
            print(f"📄 [메타데이터(출처 등)]: {metadatas[i]}")
            print(f"📝 [텍스트 내용 일부]: {documents[i][:150]} ...")
            
    except Exception as e:
        print(f"크로마 DB를 읽는 중 오류가 발생했습니다. (아직 DB가 생성 안 되었을 수 있습니다)\n상세오류: {e}")

if __name__ == "__main__":
    view_chroma_data()
