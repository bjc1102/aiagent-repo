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

def run_qa_special_indexing():
    file_path = "data/2024 알기 쉬운 의료급여제도.pdf"
    final_documents = []

    # 임베딩 모델 설정 (gemini-embedding-001)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    print(f"\n--- [Q&A 정밀 분석] 8쪽~19쪽 인덱싱 시작 ---")
    
    with pdfplumber.open(file_path) as pdf:
        # PDF 기준 8쪽~19쪽 추출 (물리적 페이지 번호로 변환 필요)
        # 보통 PDF 첫 장이 1쪽이면 index는 7부터 18까지입니다.
        # 사용자의 PDF 구조에 따라 범위를 조정하세요.
        target_pages = range(8, 19) 
        
        for p_idx in target_pages:
            try:
                page = pdf.pages[p_idx]
            except IndexError:
                break
                
            width = page.width
            
            # [핵심] 아까 성공했던 반 페이지 자르기 로직 적용
            # 왼쪽(홀수쪽), 오른쪽(짝수쪽) 영역 분할
            regions = [
                (0, 0, width / 2, page.height), 
                (width / 2, 0, width, page.height)
            ]
            
            for i, bbox in enumerate(regions):
                # 실제 페이지 번호 계산 (표기상 페이지 번호 반영)
                # 8쪽부터 시작하므로 인덱스에 따른 계산
                curr_p_label = p_idx + 1 # 단순 페이지 라벨링
                crop = page.within_bbox(bbox)
                
                # 텍스트 추출 (Q&A는 텍스트 기반 추출이 가장 정확함)
                text = crop.extract_text()
                
                if text and len(text.strip()) > 20: # 의미 있는 텍스트만 저장
                    # Q&A 가독성을 위해 문단 정리
                    clean_text = text.replace("\n", " ").strip()
                    
                    final_documents.append(Document(
                        page_content=f"### {curr_p_label}쪽 의료급여 Q&A 섹션\n{clean_text}",
                        metadata={"page": curr_p_label, "type": "qa"}
                    ))
            
            print(f"✅ PDF {p_idx + 1}쪽 (양면) 분석 완료")

    # 2. 벡터 저장소 생성 및 저장
    print(f"\n--- [임베딩] {len(final_documents)}개의 청크를 벡터화 중... ---")
    vectorstore = FAISS.from_documents(final_documents, embeddings)
    vectorstore.save_local("gemini_medical_index")
    
    # 3. 검수용 파일 생성 (qa_review.txt)
    with open("qa_review.txt", "w", encoding="utf-8") as f:
        for i, doc in enumerate(final_documents):
            f.write(f"\n{'='*60}\n")
            f.write(f"[CHUNK #{i+1:03d}] PAGE: {doc.metadata['page']}\n")
            f.write(f"{'-'*60}\n")
            f.write(f"{doc.page_content}\n")
            f.write(f"{'='*60}\n")

    print(f"\n🚀 Q&A 인덱싱 완료! 'qa_review.txt'에서 정제된 내용을 확인하세요.")

if __name__ == "__main__":
    run_qa_special_indexing()