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

def universal_table_cleaner(raw_table):
    """
    범용 표 정제 알고리즘:
    1. 첫 행을 헤더(제목)로 인식
    2. 비어있는 셀(None, '-')은 위쪽 행의 데이터를 자동으로 상속
    3. [헤더: 데이터] 형태로 결합하여 완벽한 문장형 청크 생성
    """
    if not raw_table or len(raw_table) < 2:
        return ""

    # 헤더 정리 (줄바꿈 제거 및 공백 정리)
    headers = [str(c).replace("\n", " ").strip() if c else f"항목_{i}" for i, c in enumerate(raw_table[0])]
    
    # 데이터 상속을 위한 메모리 초기화
    last_seen_values = [None] * len(headers)
    clean_rows = []

    # 데이터 행 처리 (두 번째 줄부터)
    for row_idx, row in enumerate(raw_table[1:]):
        processed_row = []
        for col_idx, cell in enumerate(row):
            # 열 개수가 헤더보다 많을 경우를 대비한 안전장치
            if col_idx >= len(headers): break
            
            val = str(cell).replace("\n", " ").strip() if cell else ""
            
            # [상속 법칙] 값이 없으면 위쪽 데이터 가져오기
            if not val or val.lower() == "none" or val == "-":
                val = last_seen_values[col_idx] if last_seen_values[col_idx] else "-"
            
            processed_row.append(val)
            last_seen_values[col_idx] = val # 다음 행을 위해 저장
        
        # [문맥 결합] "헤더명: 데이터값" 형태로 연결
        row_parts = []
        for h, v in zip(headers, processed_row):
            row_parts.append(f"{h}: {v}")
        
        clean_rows.append(" | ".join(row_parts))
        
    return "\n".join(clean_rows)

def run_fixed_width_indexing():
    file_path = "data/2024 알기 쉬운 의료급여제도 (1).pdf"
    final_documents = []

    with pdfplumber.open(file_path) as pdf:
        target_pages = [3, 4] # 5~8쪽
        
        for p_idx in target_pages:
            page = pdf.pages[p_idx]
            width = page.width
            
            # 왼쪽(0 ~ width/2), 오른쪽(width/2 ~ width) 영역
            regions = [(0, 0, width / 2, page.height), (width / 2, 0, width, page.height)]
            
            for i, bbox in enumerate(regions):
                curr_p = p_idx * 2 + 5 + i
                crop = page.within_bbox(bbox)
                
                # --- [핵심] 가로 길이 강제 제한 알고리즘 ---
                # 영역 내에서 실제 '표'가 차지해야 할 x축 범위를 더 좁게 설정합니다.
                # 여백(Margin)을 주어 표 바깥의 텍스트가 섞이지 않게 합니다.
                margin = 0.05 * width  # 전체 페이지 너비의 5%를 여백으로 설정
                table_bbox = (bbox[0] + margin, bbox[1], bbox[2] - margin, bbox[3])
                table_crop = page.within_bbox(table_bbox)
                
                # 1. 본문 추출
                text = crop.extract_text()
                if text:
                    final_documents.append(Document(
                        page_content=f"### {curr_p}쪽 본문\n{text}", 
                        metadata={"type": "text", "page": curr_p}
                    ))

                # 2. 표 추출 (반 페이지 가로 길이를 100% 활용)
                # 'vertical_strategy': 'text'를 사용해 선이 없어도 가로 끝까지 텍스트를 추적합니다.
                tables = crop.extract_tables(table_settings={
                    "vertical_strategy": "lines",    # 선이 없어도 반 페이지 안의 텍스트 정렬 기준
                    "horizontal_strategy": "lines", # 행 구분은 명확한 가로선 기준
                    "snap_tolerance": 10,            # 텍스트와 좌표 간의 유연한 매칭
                    "intersection_tolerance": 5
                })
                
                for t in tables:
                    # 우리가 만든 범용 정제기(상속 로직 + 헤더 결합) 적용
                    structured_table = universal_table_cleaner(t)
                    if structured_table:
                        final_documents.append(Document(
                            page_content=f"### {curr_p}쪽 구조화된 표 데이터\n{structured_table}",
                            metadata={"type": "table", "page": curr_p}
                        ))
                

    # (이후 벡터화 및 저장 로직은 동일)

    # --- 벡터 저장소 생성 ---
    print(f"\n--- [벡터화] 제미나이 임베딩 생성 및 FAISS 저장 ---")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(final_documents, embeddings)
    vectorstore.save_local("gemini_medical_index")

    # --- 결과 전수 검증 리포트 ---
    print(f"\n" + "="*80)
    print(f"📊 최종 인덱싱 결과 전수 검사")
    print("="*80)
    
    with open("final_chunks_review.txt", "w", encoding="utf-8") as f:
        for i, doc in enumerate(final_documents):
            output = f"\n[CHUNK #{i+1:02d}] PAGE: {doc.metadata['page']}p | TYPE: {doc.metadata['type']}\n"
            output += f"{'-'*60}\n"
            output += f"{doc.page_content}\n"
            output += f"{'-'*60}\n"
            print(output)
            f.write(output)

    print(f"\n✅ 모든 처리가 완료되었습니다. 'final_chunks_review.txt'를 확인하세요.")

if __name__ == "__main__":
    run_fixed_width_indexing()