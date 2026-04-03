import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai

# 1. 환경 설정
load_dotenv()

def run_gemini_indexing_pipeline(file_path):
    print(f"--- 1. 문서 로딩 시작 (Gemini 모드): {file_path} ---")
    
    # PDF 로딩
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    
    # 청킹 설정 (제미나이는 문맥 파악 능력이 좋으므로 넉넉하게 잡습니다)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        
        chunk_overlap=200,      
        add_start_index=True,
        separators=["\n\n", "\n", ".", " "]
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"청킹 완료. 총 청크 수: {len(splits)}")

    # 2. 제미나이 임베딩 모델 설정
    # 모델명 'models/text-embedding-004'는 현재 가장 성능이 좋은 구글 임베딩 모델입니다.
    print("--- 2. 제미나이 임베딩 진행 중... ---")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 벡터 저장소 생성 (FAISS)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    # 로컬 저장
    index_name = "gemini_medical_index"
    vectorstore.save_local(index_name)
    
    print(f"--- 3. 저장 완료: {index_name} ---")
    
    # 샘플 출력
    print("\n[제미나이 인덱싱 샘플]")
    print(f"첫 번째 청크 내용: {splits[0].page_content[:150]}...")
    
    return vectorstore

if __name__ == "__main__":
    pdf_path = "data/2024 알기 쉬운 의료급여제도.pdf"
    
    if os.path.exists(pdf_path):
        # 1. 인덱싱 실행
        vs = run_gemini_indexing_pipeline(pdf_path)
        
        # 2. Golden Dataset 기반 검색 품질 검증 (예시 질문들)
        test_questions = [
            "65세 이상 1종 수급권자 틀니 본인부담률은?",
            "2종 수급권자가 상급종합병원 외래 이용 시 본인부담금은?",
            "2종 수급권자인 3세 아동의 입원 본인부담률은?"
        ]

        print("\n" + "="*60)
        print("🔍 [Step 3] RAG 검색 품질 검증 테스트 시작")
        print("="*60)

        for j, query in enumerate(test_questions):
            print(f"\n👉 Q{j+1}. 질문: {query}")
            print(f"{'-'*60}")
            
            # 검색 수행 (Top-K = 2)
            results = vs.similarity_search(query, k=2)
            
            for i, res in enumerate(results):
                # 검색된 청크의 앞부분 300자만 깔끔하게 출력
                content = res.page_content.replace("\n", " ").strip()
                print(f"✅ 검색 결과 {i+1} (근거 조각):")
                print(f"   > {content[:300]}...")
                print(f"{'.' * 60}")
        
        print("\n" + "="*60)
        print("🎯 모든 테스트 검색이 완료되었습니다.")
        print("="*60)
    else:
        print("❌ [오류] PDF 파일을 찾을 수 없습니다. 경로를 확인해주세요.")