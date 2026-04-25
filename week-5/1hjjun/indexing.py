"""5주차 재작업: 청크 크기 재구성 (RecursiveCharacterTextSplitter)

교수님 피드백 (a) 반영:
  - 기존: 26청크 (섹션 단위, 평균 6,500자) → Recall이 0.85로 동률·Precision 역전 원인
  - 변경: chunk_size=800, chunk_overlap=100 → 100~150청크로 분산
  - HTML 태그를 텍스트로 변환 후 sub-split

흐름:
  PDF → Upstage 파싱 → 섹션별 텍스트 추출 (메타데이터 보존) →
  RecursiveCharacterTextSplitter로 800자 단위 sub-chunk → FAISS 저장
"""

import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_with_upstage(file_path):
    """Upstage Document Parse API 호출 → HTML 요소 리스트."""
    api_key = os.getenv("UPSTAGE_API_KEY")
    url = "https://api.upstage.ai/v1/document-digitization"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"document": open(file_path, "rb")}
    data = {"model": "document-parse", "ocr": "force"}
    print(f"🚀 Upstage 파싱: {os.path.basename(file_path)}")
    return requests.post(url, headers=headers, files=files, data=data).json()


def html_to_text(html: str) -> str:
    """HTML을 plain text로 변환. 표 구조 보존을 위해 줄바꿈 separator 사용."""
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n", strip=True)


def build_section_documents(upstage_result, source_year):
    """폰트 22px 헤더 기준으로 섹션 단위 Document 생성 (메타데이터 보존)."""
    elements = upstage_result.get("elements", [])
    sections = []
    cur_html = ""
    cur_meta = {"source_year": source_year, "page": 1, "section_title": "시작 섹션"}

    for el in elements:
        html = el.get("content", {}).get("html", "")
        if "data-category='paragraph' style='font-size:22px'>0" in html:
            if cur_html.strip():
                sections.append(Document(page_content=cur_html, metadata=cur_meta.copy()))
            cur_html = html
            cur_meta["page"] = el.get("page", cur_meta["page"])
            cur_meta["section_title"] = el.get("content", {}).get("text", "소주제")
        else:
            cur_html += f"\n{html}"
    if cur_html.strip():
        sections.append(Document(page_content=cur_html, metadata=cur_meta.copy()))
    return sections


def run_indexing():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    DATA_DIR = os.path.join(SCRIPT_DIR, "../../week-4/data")
    files = [
        {"year": "2025", "path": os.path.join(DATA_DIR, "2025 알기 쉬운 의료급여제도.pdf")},
        {"year": "2026", "path": os.path.join(DATA_DIR, "2026 알기 쉬운 의료급여제도.pdf")},
    ]

    all_chunks = []
    for item in files:
        if not os.path.exists(item["path"]):
            print(f"⚠️ 파일 없음: {item['path']}")
            continue

        parse_result = parse_with_upstage(item["path"])
        section_docs = build_section_documents(parse_result, item["year"])
        print(f"  📑 {item['year']}: 섹션 {len(section_docs)}개")

        # 각 섹션을 plain text로 변환 후 sub-split
        for sec in section_docs:
            clean_text = html_to_text(sec.page_content)
            sub_docs = splitter.create_documents(
                [clean_text],
                metadatas=[sec.metadata],
            )
            # chunk_id 부여
            for i, sd in enumerate(sub_docs):
                sd.metadata = {**sec.metadata, "sub_chunk_id": i}
            all_chunks.extend(sub_docs)

        # 통계
        year_chunks = [c for c in all_chunks if c.metadata.get("source_year") == item["year"]]
        avg_len = sum(len(c.page_content) for c in year_chunks) / max(1, len(year_chunks))
        print(f"  📦 {item['year']}: sub-chunk {len(year_chunks)}개, 평균 길이 {avg_len:.0f}자")

    INDEX_PATH = os.path.join(SCRIPT_DIR, "medical_advanced_index")
    print(f"\n💾 총 {len(all_chunks)}개 청크 인덱싱 중...")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"✅ 인덱스 생성 완료: {INDEX_PATH}")
    print(f"   평균 청크 길이: {sum(len(c.page_content) for c in all_chunks)/len(all_chunks):.0f}자")


if __name__ == "__main__":
    run_indexing()
