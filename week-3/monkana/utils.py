import requests
import json
import re
import pdfplumber
import io
from itertools import groupby
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
import os
import shutil

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

def reset_chroma_dir():
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("UPSTAGE_API_KEY")
PDF_PATH = BASE_DIR / "2024 알기 쉬운 의료급여제도.pdf"
CACHE_PATH = BASE_DIR / "document_parse_result.json"

# ── 1. pdfplumber 텍스트 파싱 유틸 ────────────────────────
HEADER_PATTERN = re.compile(r'^(\d{2})\s+(.+?)(\?)?$')
CIRCLE_PATTERN = re.compile(r'^◎\s+(.+)')
QA_PATTERN = re.compile(r'^Q(\d+)\.')
AA_PATTERN = re.compile(r'^A(\d+)\.')

def normalize(text: str) -> str:
    """공백 제거 후 비교용 정규화"""
    return re.sub(r'\s+', '', text)

def extract_all_lines(page) -> list[str]:
    words = page.extract_words()
    mid = page.width / 2

    left_words = [w for w in words if w['x0'] < mid]
    right_words = [w for w in words if w['x0'] >= mid]

    left_words.sort(key=lambda w: (round(w['top'] / 5) * 5, w['x0']))
    right_words.sort(key=lambda w: (round(w['top'] / 5) * 5, w['x0']))

    def words_to_lines(words_list):
        lines = []
        for _, group in groupby(words_list, key=lambda w: round(w['top'] / 5) * 5):
            line = ' '.join(w['text'] for w in group).strip()
            if line:
                lines.append(line)
        return lines

    return words_to_lines(left_words) + words_to_lines(right_words)

# ── 2. pdfplumber로 표 있는 페이지 감지 ───────────────────
def get_table_pages(pdf_path: Path) -> list[int]:
    table_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            if page.extract_tables():
                table_pages.append(page_num)
    return table_pages

# ── 3. 표 있는 페이지만 추출해서 새 PDF로 ────────────────
def extract_pages_as_pdf(pdf_path: Path, page_nums: list[int]):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    page_map = {}
    for new_page_num, orig_page_num in enumerate(page_nums, start=1):
        writer.add_page(reader.pages[orig_page_num - 1])
        page_map[new_page_num] = orig_page_num
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)
    return buf, page_map

# ── 4. Upstage API 호출 ────────────────────────────────────
def call_upstage(pdf_buf: io.BytesIO, api_key: str, page_map: dict) -> dict:
    response = requests.post(
        "https://api.upstage.ai/v1/document-digitization",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"document": ("table_pages.pdf", pdf_buf, "application/pdf")},
        data={
            "model": "document-parse",
            "output_formats": '["markdown"]',
            "coordinates": "false",
        }
    )
    data = response.json()
    for el in data.get("elements", []):
        el["page"] = page_map.get(el["page"], el["page"])
    return data

# ── 5. 캐시 처리 ──────────────────────────────────────────
if CACHE_PATH.exists():
    print("캐시 불러오는 중...")
    result = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
else:
    print("표 있는 페이지 감지 중...")
    table_pages = get_table_pages(PDF_PATH)
    print(f"표 발견 페이지: {table_pages}")

    print(f"표 페이지만 추출해서 Upstage API 호출 중... ({len(table_pages)}페이지)")
    pdf_buf, page_map = extract_pages_as_pdf(PDF_PATH, table_pages)
    print(f"페이지 매핑: {page_map}")

    result = call_upstage(pdf_buf, API_KEY, page_map)
    CACHE_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"캐시 저장 완료: {CACHE_PATH}")

# ── 6. Upstage elements에서 (page, heading) → table 매핑 ──
def is_noise_table(markdown: str) -> bool:
    lines = markdown.strip().split('\n')
    if not lines:
        return True
    cells = [c.strip() for c in lines[0].split('|') if c.strip()]
    if cells and re.match(r'^\d{2}$', cells[0]):
        return True
    if len(cells) <= 1:
        return True
    return False

def clean_heading(text: str) -> str:
    return re.sub(r'^#+\s*', '', text).strip()

def build_tables_by_page_heading(result: dict) -> dict[tuple, list[str]]:
    """
    (page, normalize(heading)) → 표 리스트 매핑
    page 내에서 heading 순서대로 표를 귀속
    """
    tables = {}
    current_key = None

    for el in result["elements"]:
        category = el["category"]
        page = el["page"]
        text = clean_heading(
            el["content"].get("markdown", "") or el["content"].get("text", "")
        )

        if category in ("heading1", "paragraph") and (
            re.match(r'^\d{2}\s+.+', text) or
            (re.match(r'^◎\s+.+', text) and page == 6)
        ):
            current_key = (page, normalize(text))
            tables.setdefault(current_key, [])

        elif category == "table" and current_key:
            md = el["content"].get("markdown", "")
            if not is_noise_table(md) and page == current_key[0]:  # ← 같은 페이지 표만
                tables[current_key].append(md)

    return tables

tables_by_page_heading = build_tables_by_page_heading(result)

# ── 7. 대섹션 매핑 ────────────────────────────────────────
PARENT_SECTION_RULES = [
    (3, range(1, 4),  "1. 의료급여제도 개요"),
    (4, range(1, 3),  "2. 의료급여 절차"),
    (5, range(1, 3),  "3. 의료급여 본인일부부담금"),
    (6, range(4, 7),  "3. 의료급여 본인일부부담금"),
    (7, range(7, 10), "3. 의료급여 본인일부부담금"),
    (7, range(1, 4),  "4. 2023년 변경된 의료급여제도"),
    (8, range(4, 9),  "4. 2023년 변경된 의료급여제도"),
    (8, range(1, 4),  "5. 의료급여기준 이력관리시스템"),
]

def get_parent_section(page: int, chunk_title: str) -> str:
    try:
        num = int(chunk_title[:2])
    except ValueError:
        return "기타"

    if page == 5 and num == 3:
        if "선택의료급여기관" in chunk_title:
            return "2. 의료급여 절차"
        else:
            return "3. 의료급여 본인일부부담금"

    for p, r, section in PARENT_SECTION_RULES:
        if p == page and num in r:
            return section
    return "기타"

# ── 8. 섹션 Ⅰ 파싱 (pdfplumber 텍스트 + Upstage 표) ──────
CIRCLE_NO_TABLE = {"치매질환"}

def parse_section_1(pdf_path: Path, tables_by_page_heading: dict) -> list[dict]:
    chunks = []
    current_parent = None
    current_texts = []
    current_page = None
    current_sub = None
    sub_chunks = []

    def get_tables(page_num: int, heading: str) -> list[str]:
        key = (page_num, normalize(heading))
        return tables_by_page_heading.get(key, [])

    def flush_sub():
        if current_sub is not None:
            sub_chunks.append({
                "parent_section": current_sub["parent_section"],
                "content": current_sub["content"],
                "text": current_sub.get("text", "").strip(),
                "tables": current_sub["tables"].copy(),
                "page": current_sub["page"],
                "section": "Ⅰ. 의료급여제도",
            })

    def flush():
        nonlocal current_sub, sub_chunks
        if not current_parent:
            return
        if current_parent.startswith("06 의료급여 2종수급권자") and sub_chunks:
            flush_sub()
            chunks.extend(sub_chunks)
            sub_chunks = []
            current_sub = None
        else:
            chunks.append({
                "parent_section": get_parent_section(current_page, current_parent),
                "content": current_parent,
                "text": '\n'.join(current_texts).strip(),
                "tables": get_tables(current_page, current_parent),
                "page": current_page,
                "section": "Ⅰ. 의료급여제도",
            })

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num > 8:
                break

            lines = extract_all_lines(page)
            for line in lines:
                if not line:
                    continue
                if '알기 쉬운 의료급여제도' in line:
                    continue

                m_header = HEADER_PATTERN.match(line)
                if m_header:
                    flush()
                    current_parent = line
                    current_texts = []
                    current_page = page_num
                    current_sub = None
                    sub_chunks = []

                elif current_parent:
                    if current_parent.startswith("06 의료급여 2종수급권자"):
                        m_circle = CIRCLE_PATTERN.match(line)
                        if m_circle:
                            flush_sub()
                            sub_title = m_circle.group(1)
                            no_table = any(k in sub_title for k in CIRCLE_NO_TABLE)
                            clean_sub_title = re.sub(r'\s+', ' ', sub_title).strip()
                            circle_heading = f"◎ {clean_sub_title}"
                            sub_table = get_tables(page_num, circle_heading) if not no_table else []
                            current_sub = {
                                "parent_section": get_parent_section(page_num, current_parent),
                                "content": f"{current_parent} > ◎ {sub_title}",
                                "text": "",
                                "tables": sub_table,
                                "page": page_num,
                                "section": "Ⅰ. 의료급여제도",
                            }
                        elif line.startswith("※") and current_sub:
                            current_sub["text"] += f"\n{line}"
                    else:
                        current_texts.append(line)

    flush()
    return chunks

# ── 9. Q&A 파싱 (pdfplumber 텍스트) ──────────────────────
def get_qa_section(q_num: int) -> str:
    if q_num <= 5:    return "1. 의료급여기준"
    elif q_num <= 22: return "2. 의료급여 절차"
    elif q_num <= 28: return "3. 선택의료급여기관 이용절차"
    elif q_num <= 33: return "4. 수가 기준 및 청구방법"
    elif q_num <= 44: return "5. 의료급여 정신질환"
    elif q_num <= 53: return "6. 본인일부부담금 적용"
    else:             return "7. 경증질환 약제비 본인부담 차등제"

def parse_section_2(pdf_path: Path) -> list[dict]:
    chunks = []
    current_q = None
    current_q_text = []
    current_a_text = []
    current_page = None
    state = None

    def flush():
        if current_q and current_a_text:
            m = QA_PATTERN.match(current_q)
            q_num = int(m.group(1)) if m else 0
            chunks.append({
                "parent_section": get_qa_section(q_num),
                "content": ' '.join(current_q_text).strip(),
                "text": ' '.join(current_a_text).strip(),
                "tables": [],
                "page": current_page,
                "section": "Ⅱ. 자주하는 질문",
            })

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num < 9:
                continue

            lines = extract_all_lines(page)
            for line in lines:
                if not line:
                    continue
                if '알기 쉬운 의료급여제도' in line:
                    continue

                if QA_PATTERN.match(line):
                    flush()
                    current_q = line
                    current_q_text = [line]
                    current_a_text = []
                    current_page = page_num
                    state = "Q"

                elif AA_PATTERN.match(line):
                    state = "A"

                elif state == "Q":
                    current_q_text.append(line)
                elif state == "A":
                    current_a_text.append(line)

    flush()
    return chunks

# ── 10. 실행 및 저장 ───────────────────────────────────────
chunks = parse_section_1(PDF_PATH, tables_by_page_heading)
qa_chunks = parse_section_2(PDF_PATH)
all_chunks = chunks + qa_chunks

print(f"섹션 Ⅰ: {len(chunks)}개")
print(f"섹션 Ⅱ: {len(qa_chunks)}개")
print(f"전체: {len(all_chunks)}개")


output_path = BASE_DIR / "all_chunks.json"
output_path.write_text(
    json.dumps(all_chunks, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"\n저장 완료: {output_path}")