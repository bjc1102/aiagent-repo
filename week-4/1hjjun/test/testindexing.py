import os
import json
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

def parse_with_upstage(file_path, output_json_path):
		"""3주차 로직: Upstage API 호출을 통해 전체 HTML 데이터 획득"""
		api_key = os.getenv("UPSTAGE_API_KEY")
		url = "https://api.upstage.ai/v1/document-digitization"
		headers = {"Authorization": f"Bearer {api_key}"}
		
		# --- 추가된 부분: 폴더가 없으면 생성 ---
		output_dir = os.path.dirname(output_json_path)
		if output_dir and not os.path.exists(output_dir):
				os.makedirs(output_dir)
				print(f"📁 폴더 생성 완료: {output_dir}")

		files = {"document": open(file_path, "rb")}
		data = {"model": "document-parse", "ocr": "force"}
		
		print(f"🚀 Upstage 파싱 시작: {os.path.basename(file_path)}")
		response = requests.post(url, headers=headers, files=files, data=data)
		
		if response.status_code == 200:
			result = response.json()
			# JSON 파일로 예쁘게 저장
			with open(output_json_path, "w", encoding="utf-8") as f:
					json.dump(result, f, ensure_ascii=False, indent=4)
			print(f"✅ 저장 완료: {output_json_path}")
			return result
		else:
			print(f"❌ 에러 발생: {response.status_code}")
			print(response.text)
			return None

def semantic_chunking_by_font(upstage_result, source_year):
    """폰트 크기 22px를 기준으로 소주제 단위 청킹 수행"""
    elements = upstage_result.get("elements", [])
    chunks = []
    
    current_chunk_content = ""
    current_metadata = {
        "source_year": source_year,
        "page": 1,
        "section_title": "시작 섹션"
    }

    for element in elements:
        html_content = element.get("content", {}).get("html", "")
        # 폰트 22px가 포함된 요소를 만나면 섹션 분할
        if "font-size:22px" in html_content:
            # 이전까지 쌓인 내용이 있다면 Document로 저장
            if current_chunk_content.strip():
                chunks.append(Document(
                    page_content=current_chunk_content.strip(),
                    metadata=current_metadata.copy()
                ))
            
            # 상태 초기화 및 새로운 섹션 정보 설정
            current_chunk_content = html_content
            current_metadata["page"] = element.get("page")
            current_metadata["section_title"] = element.get("content", {}).get("text", "소주제")
        else:
            # 일반 텍스트나 표는 현재 섹션에 계속 추가
            current_chunk_content += f"\n{html_content}"

    # 마지막 섹션 추가
    if current_chunk_content.strip():
        chunks.append(Document(
            page_content=current_chunk_content.strip(),
            metadata=current_metadata
        ))
    
    return chunks

def run_indexing():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    all_final_chunks = []
    
    # 2025/2026 PDF 파일 리스트 (경로는 본인 환경에 맞게 수정)
    target_files = [
        {"year": "2025", "path": "../data/2025 알기 쉬운 의료급여제도.pdf", "output": "2025_output.json"},
        {"year": "2026", "path": "../data/2026 알기 쉬운 의료급여제도.pdf", "output": "2026_output.json"}
    ]

    for file in target_files:
        if os.path.exists(file["path"]):
            parse_with_upstage(file["path"], file["output"])
        else:
            print(f"⚠️ 파일을 찾을 수 없습니다: {file['path']}")
            
        # 1. 파싱
        parse_result = parse_with_upstage(file["path"])
        
        # # 2. 소주제 기반 청킹
        # print(f"📦 {file['year']}년도 문서 소주제 분할 중...")
        # section_chunks = semantic_chunking_by_font(parse_result, file["year"])
        # all_final_chunks.extend(section_chunks)

    # # 3. 벡터 DB 저장
    # print(f"💾 총 {len(all_final_chunks)}개의 소주제 섹션 인덱싱 중...")
    # vectorstore = FAISS.from_documents(all_final_chunks, embeddings)
    # vectorstore.save_local("medical_advanced_index")
    # print("✅ 인덱스 생성 완료: medical_advanced_index")

if __name__ == "__main__":
    run_indexing()