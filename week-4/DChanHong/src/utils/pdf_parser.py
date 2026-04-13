import pdfplumber
import pandas as pd
from typing import List, Dict, Any
from langchain.schema import Document
import os
import re

class PDFTableParser:
    def __init__(self):
        pass

    def table_to_markdown(self, table_data: List[List[Any]]) -> str:
        """추출된 표 데이터를 마크다운 형식의 문자열로 변환합니다."""
        if not table_data or not any(table_data):
            return ""
        
        # 데이터프레임으로 변환 (첫 번째 행을 헤더로 사용 시도)
        try:
            df = pd.DataFrame(table_data[1:], columns=table_data[0])
            # 빈 셀 처리
            df = df.fillna("")
            return "\n" + df.to_markdown(index=False) + "\n"
        except Exception:
            # 헤더 처리가 실패할 경우 단순 리스트로 처리
            try:
                df = pd.DataFrame(table_data)
                df = df.fillna("")
                return "\n" + df.to_markdown(index=False, header=False) + "\n"
            except:
                return ""

    def parse_pdf(self, file_path: str) -> List[Document]:
        """PDF 파일을 읽어 텍스트와 마크다운 표가 결합된 Document 리스트를 반환합니다."""
        documents = []
        file_name = os.path.basename(file_path)
        
        # 파일명에서 연도 추출 (예: 2025 알기 쉬운... -> 2025)
        year_match = re.search(r"202\d", file_name)
        source_year = year_match.group() if year_match else "unknown"

        print(f"[{source_year}] 파싱 시작: {file_name}")

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # 1. 페이지 전체 텍스트 추출
                text = page.extract_text() or ""
                
                # 2. 페이지 내 표 추출 및 마크다운 변환
                tables = page.extract_tables()
                table_markdowns = []
                for table in tables:
                    md_table = self.table_to_markdown(table)
                    if md_table.strip():
                        table_markdowns.append(md_table)
                
                # 3. 텍스트와 표 결합 (표가 있다면 텍스트 하단에 추가)
                combined_content = text
                if table_markdowns:
                    combined_content += "\n\n### [표 데이터]\n" + "\n".join(table_markdowns)

                # 4. LangChain Document 객체 생성
                if combined_content.strip():
                    doc = Document(
                        page_content=combined_content,
                        metadata={
                            "source": file_name,
                            "source_year": source_year,
                            "page": page_num + 1
                        }
                    )
                    documents.append(doc)

        print(f"[{source_year}] 파싱 완료: {len(documents)} 페이지 추출됨")
        return documents

# 싱글톤 인스턴스
pdf_parser = PDFTableParser()
