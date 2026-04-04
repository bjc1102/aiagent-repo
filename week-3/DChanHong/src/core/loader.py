import pdfplumber
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)

def table_to_markdown(table: List[List[str]]) -> str:
    """추출된 표 리스트를 마크다운 문자열로 변환합니다."""
    if not table or not any(table):
        return ""
    
    # None 제거 및 문자열 정규화
    clean_table = []
    for row in table:
        clean_row = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
        # 모든 셀이 비어있는 행은 무시
        if any(clean_row):
            clean_table.append(clean_row)
            
    if not clean_table:
        return ""

    # 마크다운 표 생성
    md = []
    # 헤더
    md.append("| " + " | ".join(clean_table[0]) + " |")
    # 구분선
    md.append("| " + " | ".join(["---"] * len(clean_table[0])) + " |")
    # 본문
    for row in clean_table[1:]:
        md.append("| " + " | ".join(row) + " |")
        
    return "\n".join(md)

def load_pdf(file_path: str) -> List[Document]:
    """
    pdfplumber를 직접 사용하여 텍스트와 표를 분리 추출합니다.
    표는 마크다운 형식으로 변환하여 독립된 청크(Document)로 반환합니다.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    logger.info(f"표 단위 감지 로드 시작: {file_path}")
    
    documents: List[Document] = []
    
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_content = []
            
            # 1. 표(Table) 추출
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                md_table = table_to_markdown(table)
                if md_table:
                    # 표는 그 자체로 하나의 독립된 문서(청크)로 취급
                    table_doc = Document(
                        page_content=f"[Table Data from Page {i+1}]\n{md_table}",
                        metadata={
                            "source": str(path),
                            "page": i + 1,
                            "type": "table",
                            "table_index": t_idx
                        }
                    )
                    documents.append(table_doc)
            
            # 2. 일반 텍스트 추출 (표와 섞이지 않게 처리 - 필요 시 레이아웃 정보 활용)
            # 여기서는 기본 텍스트 추출 후 페이지 단위로 저장
            text = page.extract_text()
            if text:
                text_doc = Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "page": i + 1,
                        "type": "text"
                    }
                )
                documents.append(text_doc)

    logger.info(f"PDF 로드 및 표 추출 완료: {len(documents)}개 파트 추출됨")
    return documents

def load_pdfs_from_directory(directory: str) -> List[Document]:
    """디렉토리 내 모든 PDF 파일을 로드합니다."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"디렉토리가 아닙니다: {directory}")

    pdf_files = list(dir_path.glob("*.pdf"))
    all_documents: List[Document] = []
    for pdf_file in pdf_files:
        try:
            docs = load_pdf(str(pdf_file))
            all_documents.extend(docs)
        except Exception as e:
            logger.error(f"PDF 로드 실패 ({pdf_file}): {e}")

    return all_documents
