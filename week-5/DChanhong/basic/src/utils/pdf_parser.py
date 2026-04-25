import os
import re
from typing import List, Any

import pdfplumber
import pandas as pd
from langchain_core.documents import Document


class PDFTableParser:
    """PDF → 텍스트 + 표(markdown) → LangChain Document 리스트"""

    def table_to_markdown(self, table_data: List[List[Any]]) -> str:
        if not table_data or not any(table_data):
            return ""
        try:
            df = pd.DataFrame(table_data[1:], columns=table_data[0]).fillna("")
            return "\n" + df.to_markdown(index=False) + "\n"
        except Exception:
            try:
                df = pd.DataFrame(table_data).fillna("")
                return "\n" + df.to_markdown(index=False, header=False) + "\n"
            except Exception:
                return ""

    def parse_pdf(self, file_path: str) -> List[Document]:
        documents: List[Document] = []
        file_name = os.path.basename(file_path)

        year_match = re.search(r"202\d", file_name)
        source_year = year_match.group() if year_match else "unknown"

        print(f"[{source_year}] 파싱 시작: {file_name}")

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""

                table_markdowns = []
                for table in page.extract_tables():
                    md_table = self.table_to_markdown(table)
                    if md_table.strip():
                        table_markdowns.append(md_table)

                combined = text
                if table_markdowns:
                    combined += "\n\n### [표 데이터]\n" + "\n".join(table_markdowns)

                if combined.strip():
                    documents.append(
                        Document(
                            page_content=combined,
                            metadata={
                                "source": file_name,
                                "source_year": source_year,
                                "page": page_num + 1,
                            },
                        )
                    )

        print(f"[{source_year}] 파싱 완료: {len(documents)} 페이지")
        return documents


pdf_parser = PDFTableParser()
