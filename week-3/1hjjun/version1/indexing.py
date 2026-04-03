import os
import pdfplumber
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. 환경 설정
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def process_visual_table(table):
    """
    AI 가이드라인 반영:
    1단계: Grid Recognition (격자화)
    2단계: Data Inheritance (병합 셀 상속)
    3단계: Semantic Mapping (문장 결합)
    """
    if not table or len(table) < 2: return ""
    
    # 데이터 상속을 위한 메모리 (각 열의 최신 유효 데이터 저장)
    col_memory = [None] * len(table[0])
    rows_md = []

    # 마크다운 헤더 생성
    header = [str(c).replace("\n", " ").strip() if c else f"Column_{i}" for i, c in enumerate(table[0])]
    rows_md.append("| " + " | ".join(header) + " |")
    rows_md.append("| " + " | ".join(["---"] * len(header)) + " |")

    # 데이터 행 처리 (2단계: 상속 알고리즘)
    for row in table[1:]:
        processed_row = []
        for i, cell in enumerate(row):
            val = str(cell).replace("\n", " ").strip() if cell else ""
            
            # [규칙] 빈칸이거나 '-' 이면 위쪽 셀의 데이터를 상속받음
            if not val or val == "-":
                val = col_memory[i] if col_memory[i] else "-"
            
            processed_row.append(val)
            col_memory[i] = val # 다음 행을 위해 현재 값을 메모리에 저장
            
        rows_md.append("| " + " | ".join(processed_row) + " |")
    
    return "\n".join(rows_md)

def run_semantic_table_indexing():
    file_path = "data/2024 알기 쉬운 의료급여제도.pdf"
    final_documents = []

    print(f"\n--- [시각적 구조 복구] 인덱싱 파이프라인 가동 ---")
    
    with pdfplumber.open(file_path) as pdf:
        # 사용자 요청 범위: 5~8쪽 (물리적 페이지 2, 3) [cite: 14]
        target_pages = [3, 4] 
        
        for p_idx in target_pages:
            page = pdf.pages[p_idx]
            width = page.width
            
            # 4단계 가이드 중 1단계: 2단 구성 좌표 분할
            regions = [
                (0, 0, width / 2, page.height), # 왼쪽 (홀수 쪽)
                (width / 2, 0, width, page.height) # 오른쪽 (짝수 쪽)
            ]
            
            for i, bbox in enumerate(regions):
                curr_p = p_idx * 2 + 5 + i
                crop = page.within_bbox(bbox)
                
                # 본문 추출 [cite: 22]
                text = crop.extract_text()
                if text:
                    final_documents.append(Document(
                        page_content=f"### {curr_p}쪽 본문 가이드\n{text}",
                        metadata={"page": curr_p, "type": "text"}
                    ))
                
                # 표 추출 (3단계: 문맥적 관계 연결)
                tables = crop.extract_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines"
                })
                
                for t in tables:
                    md_table = process_visual_table(t)
                    if md_table:
                        final_documents.append(Document(
                            page_content=f"### {curr_p}쪽 구조화된 표 데이터\n{md_table}",
                            metadata={"page": curr_p, "type": "table"}
                        ))
                print(f"✅ {curr_p}쪽: 시각적 구조 분석 완료")

    # --- 벡터화 및 저장 ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(final_documents, embeddings)
    vectorstore.save_local("gemini_medical_index")

    # --- 전수 검증 출력 ---
    print(f"\n" + "="*80)
    print(f"🔍 [최종 검증] AI가 이해할 마크다운 데이터 구조")
    print("="*80)
    for i, doc in enumerate(final_documents):
        if doc.metadata["type"] == "table":
            print(f"\n[CHUNK #{i+1:02d}] PAGE: {doc.metadata['page']}p")
            print(doc.page_content)
            print("-" * 60)

if __name__ == "__main__":
    run_semantic_table_indexing()