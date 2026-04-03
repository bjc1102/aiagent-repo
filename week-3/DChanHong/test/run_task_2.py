import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.core.loader import load_pdf
from src.core.splitter import split_documents
from src.core.embedder import create_vector_store
from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

def run_task_2():
    # 1. PDF 로딩
    pdf_path = "/Users/root1/Desktop/aiagent-repo/week-3/data/2024 알기 쉬운 의료급여제도.pdf"
    print(f"--- 1. PDF 로딩 시작: {pdf_path} ---")
    documents = load_pdf(pdf_path)
    print(f"로드 완료: {len(documents)} 페이지")

    # 2. 청킹 설정 및 실행
    print(f"\n--- 2. 청킹 시작 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ---")
    chunks = split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    total_chunks = len(chunks)
    print(f"청킹 완료: 총 {total_chunks}개 청크 생성")

    # 3. 표 데이터 확인 및 청크 샘플 추출
    print("\n--- 3. 표 데이터 포함 청크 샘플 추출 (상위 2~3개) ---")
    # 표 데이터가 포함되었을 법한 청크 찾기 (간단한 키워드나 패턴 사용)
    # 의료급여제도 문서에는 보통 테이블 형식이 많으므로 '┃', '─', '구분', '항목', '금액' 등을 찾아봅니다.
    table_patterns = ["┃", "─", "|", "구분", "내용", "대상", "금액"]
    table_chunks = []
    
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        # 간단한 휴리스틱: '|' 나 '┃' 가 포함되어 있거나 '구분' 등의 키워드가 있으면 표로 간주
        if any(pattern in content for pattern in table_patterns):
            table_chunks.append((i, content))
        if len(table_chunks) >= 3:
            break

    for i, (idx, content) in enumerate(table_chunks):
        print(f"\n[표 포함 청크 샘플 {i+1} (Index: {idx})]")
        print("-" * 40)
        # 내용이 길 수 있으므로 일부만 출력
        print(content[:500] + "...")
        print("-" * 40)

    # 4. 임베딩 및 벡터 저장소 저장
    print(f"\n--- 4. 임베딩 및 Chroma DB 저장 시작 (모델: {EMBEDDING_MODEL}) ---")
    vector_store = create_vector_store(chunks)
    print("저장 완료!")

    # 기록용 데이터 출력
    print("\n" + "="*50)
    print("TASK 2 결과 기록")
    print("="*50)
    print(f"- PDF 로더: PyPDFLoader (langchain_community)")
    print(f"- Text Splitter: RecursiveCharacterTextSplitter")
    print(f"- chunk_size: {CHUNK_SIZE}")
    print(f"- chunk_overlap: {CHUNK_OVERLAP}")
    print(f"- 총 청크 수: {total_chunks}")
    print(f"- 임베딩 모델명: {EMBEDDING_MODEL}")
    print(f"- 벡터 저장소 종류: Chroma")
    print("="*50)

if __name__ == "__main__":
    run_task_2()
