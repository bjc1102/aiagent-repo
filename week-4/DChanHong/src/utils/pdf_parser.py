import pdfplumber
import pandas as pd
from typing import List, Dict, Any
from langchain.schema import Document
import os
import re

# =============================================================================
# [class란?]
# class는 "설계도"라고 생각하면 됩니다.
# 예를 들어, "자동차 설계도"가 있으면 그 설계도로 여러 자동차를 만들 수 있듯이,
# class로 정의해두면 그 설계도를 기반으로 "객체(인스턴스)"를 만들어 사용합니다.
#
# class 안에 정의된 함수(def)를 "메서드"라고 부릅니다.
# 메서드는 그 class가 할 수 있는 "동작/기능"입니다.
#
# [self란?]
# class 안의 모든 메서드는 첫 번째 파라미터로 self를 받습니다.
# self는 "이 객체 자기 자신"을 가리킵니다.
# 예: pdf_parser.parse_pdf("파일.pdf") 호출 시, self = pdf_parser 가 됩니다.
# =============================================================================


class PDFTableParser:
    """
    PDF 파일에서 텍스트와 표(table)를 추출하는 클래스입니다.

    이 클래스가 하는 일:
    1. PDF 파일을 열어서 각 페이지의 텍스트를 추출
    2. 페이지 안에 표(table)가 있으면 마크다운 형식으로 변환
    3. 텍스트 + 표를 합쳐서 LangChain Document 객체로 만듦
       (Document = RAG 파이프라인에서 사용하는 표준 데이터 형태)
    """

    def __init__(self):
        """
        [__init__이란?]
        클래스로 객체를 만들 때 "자동으로 처음 한 번" 실행되는 메서드입니다.
        보통 여기서 초기 설정(변수 초기화 등)을 합니다.

        예시:
            parser = PDFTableParser()  # 이 순간 __init__이 자동 실행됨

        여기서는 특별히 초기화할 것이 없어서 pass (아무것도 안 함)
        """
        pass

    def table_to_markdown(self, table_data: List[List[Any]]) -> str:
        """
        추출된 표 데이터를 마크다운 형식의 문자열로 변환합니다.

        [파라미터 설명]
        - table_data: 2차원 리스트 (리스트 안에 리스트)
          예시: [["이름", "나이"], ["홍길동", "30"], ["김철수", "25"]]
                 ^^^^^^^^^^^^^^^^  <- 첫 번째 행 = 헤더(컬럼명)
                                    ^^^^^^^^^^^^^^^^  <- 두 번째 행부터 = 데이터

        [타입 힌트 설명]
        - List[List[Any]]: "Any 타입의 값을 담은 리스트"의 리스트
          -> 쉽게 말해 "2차원 배열"
        - -> str: 이 함수가 리턴하는 값의 타입이 문자열(str)이라는 뜻
          -> 타입 힌트는 강제가 아니라 "이렇게 쓸 거예요"라는 안내표 같은 것

        [리턴값]
        마크다운 표 문자열. 예:
        | 이름   | 나이 |
        |--------|------|
        | 홍길동 | 30   |
        | 김철수 | 25   |
        """

        # table_data가 비어있거나, 모든 행이 비어있으면 빈 문자열 반환
        # any()는 하나라도 True(=값이 있는)인 게 있으면 True를 리턴
        if not table_data or not any(table_data):
            return ""

        # pandas DataFrame으로 변환하여 마크다운 표로 만듦
        # DataFrame = 엑셀의 시트 같은 2차원 표 데이터 구조
        try:
            # table_data[0]  -> 첫 번째 행을 컬럼명(헤더)으로 사용
            # table_data[1:] -> 두 번째 행부터 끝까지를 데이터로 사용
            # [1:]은 "슬라이싱" 문법: 인덱스 1번부터 끝까지 잘라냄
            df = pd.DataFrame(table_data[1:], columns=table_data[0])

            # fillna("") -> NaN(빈 값)을 빈 문자열("")로 채움
            # NaN = "Not a Number", pandas에서 "값이 없음"을 표현하는 방법
            df = df.fillna("")

            # to_markdown() -> DataFrame을 마크다운 표 문자열로 변환
            # index=False -> 왼쪽에 행 번호(0, 1, 2...) 안 붙임
            return "\n" + df.to_markdown(index=False) + "\n"
        except Exception:
            # 위 방법이 실패하면 (예: 헤더 개수와 데이터 개수가 안 맞을 때)
            # 헤더 없이 전체를 그냥 데이터로 처리
            try:
                df = pd.DataFrame(table_data)
                df = df.fillna("")
                return "\n" + df.to_markdown(index=False, header=False) + "\n"
            except:
                # 그래도 실패하면 빈 문자열 반환 (표를 포기)
                return ""

    def parse_pdf(self, file_path: str) -> List[Document]:
        """
        PDF 파일을 읽어서 텍스트와 표를 추출한 뒤,
        LangChain Document 리스트로 반환합니다.

        [파라미터]
        - file_path: PDF 파일의 경로 (예: "../data/2025_세법.pdf")

        [리턴값]
        - List[Document]: Document 객체들의 리스트
          각 Document는 PDF의 한 페이지에 해당하며, 다음을 담고 있음:
            - page_content: 그 페이지의 텍스트 + 표 내용
            - metadata: 부가 정보 (파일명, 연도, 페이지 번호)
        """

        # 결과를 담을 빈 리스트. 여기에 Document를 하나씩 추가(append)할 예정
        documents = []

        # os.path.basename: 전체 경로에서 파일명만 추출
        # 예: "../data/2025_세법.pdf" -> "2025_세법.pdf"
        file_name = os.path.basename(file_path)

        # 파일명에서 연도(2020~2029) 추출
        # re.search(패턴, 문자열) -> 정규표현식으로 문자열에서 패턴을 찾음
        # r"202\d" -> "202" 다음에 숫자(0-9) 하나가 오는 패턴
        # 예: "2025 알기 쉬운..." -> "2025" 매칭됨
        year_match = re.search(r"202\d", file_name)

        # 매칭 결과가 있으면 .group()으로 매칭된 문자열을 가져옴
        # 없으면 "unknown"
        source_year = year_match.group() if year_match else "unknown"

        # f-string: 문자열 안에 {변수}를 넣어서 값을 삽입하는 방법
        print(f"[{source_year}] 파싱 시작: {file_name}")

        # =====================================================================
        # pdfplumber.open()으로 PDF 파일을 열어서 처리
        #
        # [with 문이란?]
        # 파일을 열면 반드시 닫아야 하는데, with 문을 사용하면
        # 블록이 끝날 때 자동으로 파일을 닫아줌 (실수로 안 닫는 걸 방지)
        #
        # with A as B:  -> A를 실행한 결과를 B라는 변수에 담음
        # =====================================================================
        with pdfplumber.open(file_path) as pdf:

            # enumerate()는 리스트를 순회하면서 (인덱스, 값) 쌍을 줌
            # 예: enumerate(["a","b","c"]) -> (0,"a"), (1,"b"), (2,"c")
            # 여기서 page_num = 인덱스(0부터), page = 각 페이지 객체
            for page_num, page in enumerate(pdf.pages):

                # ----- 단계 1: 페이지 전체 텍스트 추출 -----
                # page.extract_text()가 None을 리턴할 수 있으므로
                # "or """로 None이면 빈 문자열을 대신 사용
                text = page.extract_text() or ""

                # ----- 단계 2: 페이지 안의 표(table)들을 추출 -----
                # extract_tables()는 표가 여러 개일 수 있으므로 리스트로 리턴
                # 예: tables = [ [[표1 데이터]], [[표2 데이터]] ]
                tables = page.extract_tables()

                # 각 표를 마크다운으로 변환한 결과를 담을 리스트
                table_markdowns = []
                for table in tables:
                    # self.table_to_markdown(table)
                    # -> 위에서 정의한 table_to_markdown 메서드를 호출
                    # self.메서드() = 같은 클래스 안의 다른 메서드를 호출하는 방법
                    md_table = self.table_to_markdown(table)

                    # .strip()은 문자열 앞뒤 공백/줄바꿈을 제거
                    # 내용이 있는 표만 리스트에 추가
                    if md_table.strip():
                        table_markdowns.append(md_table)

                # ----- 단계 3: 텍스트와 표를 하나로 합침 -----
                combined_content = text
                if table_markdowns:
                    # 표가 있으면 텍스트 아래에 "[표 데이터]" 섹션으로 추가
                    # "\n".join(리스트) -> 리스트의 각 요소를 줄바꿈으로 이어붙임
                    combined_content += "\n\n### [표 데이터]\n" + "\n".join(table_markdowns)

                # ----- 단계 4: LangChain Document 객체 생성 -----
                # 내용이 있는 페이지만 Document로 만듦
                if combined_content.strip():
                    doc = Document(
                        # page_content: 실제 텍스트 내용 (RAG에서 검색 대상이 됨)
                        page_content=combined_content,
                        # metadata: 부가 정보 (검색 결과에서 출처 표시 등에 사용)
                        metadata={
                            "source": file_name,        # 어떤 PDF에서 왔는지
                            "source_year": source_year,  # 몇 년도 자료인지
                            "page": page_num + 1         # 몇 페이지인지 (1부터 시작)
                        }
                    )
                    # 리스트에 추가
                    # append = 리스트 맨 뒤에 요소 하나를 추가하는 메서드
                    documents.append(doc)

        print(f"[{source_year}] 파싱 완료: {len(documents)} 페이지 추출됨")

        # 모든 페이지를 처리한 Document 리스트를 리턴
        return documents


# =============================================================================
# [싱글톤 패턴이란?]
# 클래스로 객체를 딱 한 번만 만들어서 여러 곳에서 공유하는 방식입니다.
#
# 아래처럼 모듈 레벨에서 객체를 하나 만들어두면,
# 다른 파일에서 "from src.utils.pdf_parser import pdf_parser" 로
# 이미 만들어진 객체를 가져다 쓸 수 있습니다.
#
# 매번 PDFTableParser()를 새로 만들 필요 없이,
# pdf_parser 하나를 공유해서 사용하는 것입니다.
# =============================================================================
pdf_parser = PDFTableParser()
