import requests
import json
import re
import pdfplumber
import io
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
import os
import shutil

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR.parent / "data"
CHROMA_DIR = BASE_DIR.parent / "chroma_db"

def reset_chroma_dir():
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("UPSTAGE_API_KEY")
PDF_PATH = DATA_DIR / "2026 알기 쉬운 의료급여제도.pdf"
CACHE_PATH = DATA_DIR / "2026 알기 쉬운 의료급여제도.cache.json"

# ── Patterns ───────────────────────────────────────────────
MAJOR_HEADING_RE   = re.compile(r'^(\d{2})\s+(.+)')  # "01 XXX" (exactly 2 digits)
SUB_NUM_RE         = re.compile(r'^\d+$')             # "1", "2", ... (standalone)
SUB_NUM_TITLE_RE   = re.compile(r'^(\d)\s+(.+)')      # "5 XXX" (single digit + title on same line)
OCR_SUB_NUM_TITLE_RE = re.compile(r'^[bB]\s+(.+)')    # OCR often reads "6" as "b"
CIRCLE_RE          = re.compile(r'^○\s*(.*)')         # "○ XXX"
NOTE_RE            = re.compile(r'^※')                # "※ ..."
Q_RE               = re.compile(r'^Q(\d+)$')          # "Q1"
A_RE               = re.compile(r'^A\s')              # "A ..."
FOOTER_RE          = re.compile(r'^\s*(?:\d+\s*[•·°]\s*)?(?:20\d{2}\s*)?알기 쉬운 의료급여제도\s*$|^\s*Ⅰ\.\s*의료급여제도\s*(?:•\s*\d+)?\s*$')
TABLE_TITLE_KEYWORDS = (
    "수급권자란",
    "수급권자 구분",
    "의료급여기관 구분",
    "단계별 진료 예외",
    "선택의료급여기관 단계별 진료 예외",
    "의료급여기관 이용",
    "본인부담금 면제자",
    "본인부담률",
    "본인부담 기준",
)

DOCUMENT_PAGES = range(5, 36)    # 5–35
SECTION_1_PAGES = range(5, 16)   # 5–15
SECTION_2_PAGES = range(17, 36)  # 17–35


def normalize(text: str) -> str:
    return re.sub(r'\s+', '', text)


def comparable_heading(text: str) -> str:
    text = normalize(clean_heading(text))
    text = re.sub(r'^(?:\d+|○)+', '', text)
    return text


def heading_aliases(text: str) -> list[str]:
    comparable = comparable_heading(text)
    aliases = [comparable]

    if "수급권자란" in comparable:
        aliases.append(comparable_heading("수급권자 구분"))
    if "의료급여기관이란" in comparable:
        aliases.append(comparable_heading("의료급여기관 구분"))
    if "3단계의료급여절차" in comparable or "단계의료급여절차" in comparable:
        aliases.append(comparable_heading("의료급여 단계별 진료 예외 사항"))
    if "선택의료급여기관이용절차" in comparable:
        aliases.append(comparable_heading("선택의료급여기관 단계별 진료 예외사항"))
    if "의료급여기관이용시본인부담률및부담액" in comparable:
        aliases.append(comparable_heading("의료급여기관 이용 시 본인부담률 및 부담액"))

    return aliases


def is_noise(line: str) -> bool:
    return (is_footer_text(line)
            or 'Ⅱ. 자주하는 질문' in line)


def is_footer_text(text: str) -> bool:
    text = clean_heading(text).strip()
    return bool(
        "알기 쉬운 의료급여제도" in text
        or "Ⅰ. 의료급여제도" in text
        or re.match(r'^\s*I\s+의료급여제도', text)
        or
        FOOTER_RE.match(text)
        or re.match(r'^\s*\d+\s*[•·°]\s*20\d{2}\s*알기 쉬운 의료급여제도\s*$', text)
        or re.match(r'^\s*\d+\s*20\d{2}\s*알기 쉬운 의료급여제도\s*$', text)
    )


def clean_table_markdown(md: str) -> str:
    cleaned_lines = []
    for line in md.splitlines():
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if cells and any(is_footer_text(cell) for cell in cells):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def table_rows(md: str) -> list[list[str]]:
    rows = []
    for line in md.splitlines():
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if cells and not all(c == "---" for c in cells):
            rows.append(cells)
    return rows


def trim_table_markdown(md: str) -> tuple[str, str]:
    md = clean_table_markdown(md)
    lines = md.splitlines()
    heading = infer_heading_from_table(md) or ""

    start_idx = None
    for i, line in enumerate(lines):
        if "|구분|" in line.replace(" ", ""):
            start_idx = i
            break

    if start_idx is None:
        return md, heading

    kept = [lines[start_idx]]
    header_cells = [c.strip() for c in lines[start_idx].split("|") if c.strip()]
    if header_cells:
        kept.append("| " + " | ".join("---" for _ in header_cells) + " |")

    for line in lines[start_idx + 1:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if cells and all(c == "---" for c in cells):
            continue
        kept.append(line)
    return "\n".join(kept).strip(), heading


def is_value_like(text: str) -> bool:
    text = clean_heading(text).strip()
    return bool(re.match(r'^(무료|무\s*료|\d+%|\d{1,3}(?:,\d{3})*원|병원급\s*이상\s*\d+%)$', text))


def is_chunk_boundary_line(line: str) -> bool:
    line = clean_heading(line).strip()
    return bool(
        NOTE_RE.match(line)
        or MAJOR_HEADING_RE.match(line)
        or SUB_NUM_RE.match(line)
        or SUB_NUM_TITLE_RE.match(line)
        or OCR_SUB_NUM_TITLE_RE.match(line)
        or CIRCLE_RE.match(line)
        or is_sub_title_without_number(line)
    )


def is_table_spillover_fragment(line: str) -> bool:
    line = clean_heading(line).strip()
    if not line:
        return False
    if line.startswith("*"):
        return False
    if line in {"구분", "본인부담률", "본인부담액", "1종", "2종"}:
        return True
    if is_chunk_boundary_line(line):
        return False
    if is_value_like(line):
        return True
    if len(line) <= 35 and re.search(r'(디스크|협착증|질환|입원|외래|무료|급여기관|병원급|조산아|출생아)', line):
        return True
    return False


def append_table_fragment(md: str, fragment: str) -> str:
    fragment = clean_heading(fragment).strip()
    if not fragment:
        return md

    lines = md.splitlines()
    if not lines:
        return md

    if is_value_like(fragment):
        for i in range(len(lines) - 1, -1, -1):
            cells = [c.strip() for c in lines[i].split("|")[1:-1]]
            if len(cells) != 2 or "*" not in cells[0]:
                continue

            if "장기지속형 주사제*" in cells[0]:
                head, tail = cells[0].split("장기지속형 주사제*", 1)
                if head.strip():
                    lines[i] = f"| {head.strip()} | {cells[1]} |"
                    lines.insert(i + 1, f"| 장기지속형 주사제*{tail.strip()} | {fragment} |")
                    return "\n".join(lines)

            m = re.match(r'(.+?)\s+([^|]+?\*)$', cells[0])
            if not m:
                break

            lines[i] = f"| {m.group(1).strip()} | {cells[1]} |"
            lines.insert(i + 1, f"| {m.group(2).strip()} | {fragment} |")
            return "\n".join(lines)

    header_cells = [c.strip() for c in lines[0].split("|")[1:-1]]
    if len(header_cells) >= 2:
        return f"{md}\n| {fragment} |  |"
    return f"{md}\n{fragment}"


def is_orphan_table_header_line(line: str) -> bool:
    line = clean_heading(line).strip()
    if not line:
        return False
    if (
        NOTE_RE.match(line)
        or MAJOR_HEADING_RE.match(line)
        or SUB_NUM_RE.match(line)
        or SUB_NUM_TITLE_RE.match(line)
        or OCR_SUB_NUM_TITLE_RE.match(line)
        or CIRCLE_RE.match(line)
    ):
        return False
    if line == "구분":
        return True
    return bool(len(line) <= 40 and line.startswith("구분 "))


def prepend_table_header(md: str, header_lines: list[str]) -> str:
    if not header_lines:
        return md
    if "|구분|" in md.replace(" ", ""):
        return md

    cells = []
    for line in header_lines:
        line = clean_heading(line).strip()
        if line == "구분":
            cells.append(line)
        else:
            cells.extend(line.split())
    cells = [cell for cell in cells if cell]
    if not cells:
        return md

    body_lines = []
    for line in md.splitlines():
        row_cells = [c.strip() for c in line.split("|") if c.strip()]
        if row_cells and all(c == "---" for c in row_cells):
            continue
        body_lines.append(line)

    header = "| " + " | ".join(cells) + " |"
    sep = "| " + " | ".join("---" for _ in cells) + " |"
    return "\n".join([header, sep, *body_lines]).strip()


def clean_chunk_text(text: str) -> str:
    cut_patterns = [
        "건강보험심사평가원은 건강하고 안전한 의료문화를",
        "부정청탁 금지",
        "제재내용 제재대상",
        "2025 알기 쉬운 의료급여제도 발 행",
        "2026 알기 쉬운 의료급여제도 발 행",
    ]
    for pattern in cut_patterns:
        idx = text.find(pattern)
        if idx != -1:
            text = text[:idx]

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or is_footer_text(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def is_table_heading(text: str) -> bool:
    return bool(
        MAJOR_HEADING_RE.match(text)
        or SUB_NUM_RE.match(text)
        or SUB_NUM_TITLE_RE.match(text)
        or CIRCLE_RE.match(text)
        or any(keyword in text for keyword in TABLE_TITLE_KEYWORDS)
    )


def element_text(el: dict) -> str:
    return clean_heading(el["content"].get("markdown", "") or el["content"].get("text", "")).strip()


def split_element_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r'^\s*[-•·]\s*', '', raw).strip()
        if line:
            lines.append(line)
    return lines


def is_sub_title_without_number(line: str) -> bool:
    line = clean_heading(line).strip()
    if not line or CIRCLE_RE.match(line) or NOTE_RE.match(line):
        return False
    if line.startswith(("○", "。", "-", "•", "·", "*", "(")):
        return False
    if len(line) > 80:
        return False
    return (
        line.endswith("란?")
        or "본인부담률" in line
        or "본인부담금" in line
        or "의료급여기관 이용" in line
        or "65세 이상 등록된" in line
        or "국가건강검진" in line
        or "특수식 식대" in line
    )


def infer_heading_from_table(md: str):
    norm_md = normalize(md)
    if "의료급여기관이용" in norm_md and "본인부담률" in norm_md and "부담액" in norm_md:
        return "1 의료급여기관 이용 시 본인부담률 및 부담액"
    if "제1차의료급여기관" in norm_md and "제2차의료급여기관" in norm_md and "제3차의료급여기관" in norm_md:
        return "3 의료급여기관이란?"

    for line in md.splitlines():
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if not cells or set(cells) <= {"---"}:
            continue
        first_cell = clean_heading(cells[0])
        if SUB_NUM_TITLE_RE.match(first_cell) or CIRCLE_RE.match(first_cell):
            return first_cell
        return None
    return None


def char_in_bbox(obj: dict, bbox: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom


def extract_text_without_tables(page) -> str:
    table_bboxes = [table.bbox for table in page.find_tables()]
    if not table_bboxes:
        return page.extract_text() or ""
    filtered_page = page.filter(
        lambda obj: obj.get("object_type") != "char"
        or not any(char_in_bbox(obj, bbox) for bbox in table_bboxes)
    )
    return filtered_page.extract_text() or ""


# ── Upstage: detect & extract table pages ─────────────────
def get_table_pages(pdf_path: Path) -> list[int]:
    table_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            if i not in DOCUMENT_PAGES:
                continue
            if page.extract_tables():
                table_pages.append(i)
    return table_pages


def extract_pages_as_pdf(pdf_path: Path, page_nums: list[int]):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    page_map = {}
    for new_i, orig_i in enumerate(page_nums, start=1):
        writer.add_page(reader.pages[orig_i - 1])
        page_map[new_i] = orig_i
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)
    return buf, page_map


def call_upstage(pdf_buf: io.BytesIO, api_key: str, page_map: dict) -> dict:
    resp = requests.post(
        "https://api.upstage.ai/v1/document-digitization",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"document": ("table_pages.pdf", pdf_buf, "application/pdf")},
        data={"model": "document-parse", "output_formats": '["markdown"]', "coordinates": "false"},
    )
    data = resp.json()
    for el in data.get("elements", []):
        el["page"] = page_map.get(el["page"], el["page"])
    return data


# ── Cache ──────────────────────────────────────────────────
if CACHE_PATH.exists():
    print("캐시 불러오는 중...")
    result = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
else:
    print("표 있는 페이지 감지 중...")
    table_pages = get_table_pages(PDF_PATH)
    print(f"표 발견 페이지: {table_pages}")
    pdf_buf, page_map = extract_pages_as_pdf(PDF_PATH, table_pages)
    assert API_KEY, "UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다."
    result = call_upstage(pdf_buf, API_KEY, page_map)
    CACHE_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"캐시 저장 완료: {CACHE_PATH}")

TABLE_PAGES = set(get_table_pages(PDF_PATH))


# ── Table mapping: (page, normalized_heading) → [md] ──────
def is_noise_table(md: str) -> bool:
    md = clean_table_markdown(md)

    if "부정청탁" in md:
        return True

    rows = []
    for line in md.strip().split('\n'):
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if not cells or all(c == "---" for c in cells):
            continue
        rows.append(cells)

    if not rows:
        return True

    if re.match(r'^\d{2}$', rows[0][0]):
        return True

    return max(len(cells) for cells in rows) <= 1


def clean_heading(text: str) -> str:
    return re.sub(r'^#+\s*', '', text).strip()


def build_tables_by_page_heading(result: dict) -> dict:
    tables = {}
    current_key = None
    for el in result["elements"]:
        category = el["category"]
        page     = el["page"]
        text     = clean_heading(el["content"].get("markdown", "") or el["content"].get("text", ""))

        if category in ("heading1", "heading2", "paragraph"):
            if is_table_heading(text):
                current_key = (page, normalize(text))
                tables.setdefault(current_key, [])

        elif category == "table":
            md = el["content"].get("markdown", "")
            if is_noise_table(md):
                continue
            inferred_heading = infer_heading_from_table(md)
            md = clean_table_markdown(md)
            if inferred_heading:
                key = (page, normalize(inferred_heading))
            elif current_key and page == current_key[0]:
                key = current_key
            else:
                continue
            tables.setdefault(key, []).append(md)

    return tables


tables_by_page_heading = build_tables_by_page_heading(result)


def build_tables_by_page(result: dict, pages: range) -> dict[int, list[str]]:
    tables = {}
    for el in result["elements"]:
        if el["category"] != "table" or el["page"] not in pages:
            continue
        md = el["content"].get("markdown", "")
        if is_noise_table(md):
            continue
        md = clean_table_markdown(md)
        tables.setdefault(el["page"], []).append(md)
    return tables


section_2_tables_by_page = build_tables_by_page(result, SECTION_2_PAGES)


def get_tables_for(page_num: int, heading: str) -> list[str]:
    norm = normalize(heading)
    comparables = heading_aliases(heading)
    key = (page_num, norm)
    if key in tables_by_page_heading and tables_by_page_heading[key]:
        return tables_by_page_heading[key]
    for (p, k), v in tables_by_page_heading.items():
        if p != page_num or not v:
            continue
        table_comparable = comparable_heading(k)
        for comparable in comparables:
            if comparable and table_comparable and (
                comparable == table_comparable
                or comparable in table_comparable
                or table_comparable in comparable
            ):
                return v
    # ○ circle: partial/substring match (handles Upstage-split headings)
    if norm.startswith("○") and len(norm) > 3:
        for (p, k), v in tables_by_page_heading.items():
            if p == page_num and v and (norm in k or k in norm):
                return v
    # digit-starting (sub_num + title): prefix match (handles truncated pdfplumber titles)
    if re.match(r'^\d', norm) and len(norm) > 1:
        for (p, k), v in tables_by_page_heading.items():
            if p == page_num and v and (k.startswith(norm) or norm.startswith(k)):
                return v
    return []


# ── Section I parsing (pages 5–15) ────────────────────────
def parse_section_1_hybrid(pdf_path: Path, table_pages: set[int]) -> list[dict]:
    chunks = []
    upstage_by_page = {}
    for el in result.get("elements", []):
        page = el.get("page")
        if page in SECTION_1_PAGES:
            upstage_by_page.setdefault(page, []).append(el)

    current_major = None
    current_sub_num = None
    current_sub_title = None
    current_texts = []
    current_tables = []
    current_page = None
    awaiting_title = False
    in_section_9 = False
    circle_title = None
    circle_texts = []
    circle_tables = []
    circle_page = None
    pending_table_idx = None
    pending_table_header_lines = []

    def current_major_num():
        if not current_major:
            return ""
        m = MAJOR_HEADING_RE.match(current_major)
        return m.group(1) if m else ""

    def next_sub_num():
        if current_sub_num and current_sub_num.isdigit():
            return str(int(current_sub_num) + 1)
        return "1"

    def flush_circle():
        nonlocal circle_title, circle_texts, circle_tables, circle_page, pending_table_idx
        if circle_title is None:
            return
        chunks.append({
            "source": "2026",
            "section": "Ⅰ. 의료급여제도",
            "parent_subject": current_major,
            "subject": f"{current_sub_num} {current_sub_title} > {circle_title}".strip(),
            "text": clean_chunk_text("\n".join(circle_texts)),
            "tables": circle_tables.copy(),
            "page": circle_page,
        })
        circle_title = None
        circle_texts = []
        circle_tables = []
        circle_page = None
        pending_table_idx = None

    def flush_sub():
        nonlocal pending_table_idx
        if current_sub_num is None or current_major is None or in_section_9:
            return
        chunks.append({
            "source": "2026",
            "section": "Ⅰ. 의료급여제도",
            "parent_subject": current_major,
            "subject": f"{current_sub_num} {current_sub_title or ''}".strip(),
            "text": clean_chunk_text("\n".join(current_texts)),
            "tables": current_tables.copy(),
            "page": current_page,
        })
        pending_table_idx = None

    def flush_current():
        if in_section_9:
            flush_circle()
        else:
            flush_sub()

    def start_sub(num: str, title, page_num: int):
        nonlocal current_sub_num, current_sub_title, current_texts, current_tables
        nonlocal current_page, awaiting_title, in_section_9, circle_title, circle_texts, circle_tables, circle_page
        flush_current()
        pending_table_header_lines.clear()
        in_section_9 = (current_major_num() == "03" and num == "9")
        current_sub_num = num
        current_sub_title = title
        current_texts = []
        current_tables = []
        current_page = page_num
        awaiting_title = (title is None)
        circle_title = None
        circle_texts = []
        circle_tables = []
        circle_page = None

    def append_text(line: str, page_num: int):
        nonlocal current_sub_title, current_tables, awaiting_title
        nonlocal circle_title, circle_texts, circle_tables, circle_page, pending_table_idx
        line = clean_heading(line).strip()
        if not line or is_noise(line):
            return
        if pending_table_idx is not None and is_table_spillover_fragment(line):
            if line in {"구분", "본인부담률", "본인부담액"}:
                return
            if in_section_9 and circle_tables:
                circle_tables[pending_table_idx] = append_table_fragment(circle_tables[pending_table_idx], line)
            elif current_tables:
                current_tables[pending_table_idx] = append_table_fragment(current_tables[pending_table_idx], line)
            return
        pending_table_idx = None

        if current_major is None:
            return

        if awaiting_title:
            awaiting_title = False
            if not CIRCLE_RE.match(line) and not NOTE_RE.match(line):
                current_sub_title = line
                return

        if is_orphan_table_header_line(line):
            pending_table_header_lines.append(line)
            return

        if not in_section_9:
            if SUB_NUM_RE.match(line):
                start_sub(line, None, page_num)
                return
            m_snt = SUB_NUM_TITLE_RE.match(line)
            if m_snt:
                start_sub(m_snt.group(1), m_snt.group(2), page_num)
                return
            m_ocr = OCR_SUB_NUM_TITLE_RE.match(line)
            if m_ocr:
                start_sub("6", m_ocr.group(1), page_num)
                return
            if is_sub_title_without_number(line):
                start_sub(next_sub_num(), line, page_num)
                return

        m_c = CIRCLE_RE.match(line)
        if m_c:
            content = m_c.group(1).strip()
            if in_section_9:
                flush_circle()
                circle_title = content
                circle_texts = []
                circle_tables = []
                circle_page = page_num
            else:
                    current_texts.append(line)
            return

        if in_section_9 and circle_title and not circle_texts and not circle_tables:
            if circle_title.endswith(("및", ",")):
                circle_title = f"{circle_title} {line}".strip()
                return

        if in_section_9 and circle_title:
            circle_texts.append(line)
        else:
            current_texts.append(line)

    nonlocal_vars = {}

    def set_major(line: str):
        nonlocal current_major, current_sub_num, current_sub_title, current_texts, current_tables
        nonlocal current_page, awaiting_title, in_section_9, circle_title, circle_texts, circle_tables, circle_page, pending_table_idx
        flush_current()
        pending_table_header_lines.clear()
        current_major = line
        current_sub_num = None
        current_sub_title = None
        current_texts = []
        current_tables = []
        current_page = None
        awaiting_title = False
        in_section_9 = False
        circle_title = None
        circle_texts = []
        circle_tables = []
        circle_page = None
        pending_table_idx = None

    def handle_text_line(line: str, page_num: int):
        if MAJOR_HEADING_RE.match(clean_heading(line).strip()):
            set_major(clean_heading(line).strip())
        elif SUB_NUM_RE.match(clean_heading(line).strip()) and len(clean_heading(line).strip()) == 2:
            set_major(clean_heading(line).strip())
            nonlocal_vars["awaiting_major_title"] = True
        elif nonlocal_vars.get("awaiting_major_title"):
            nonlocal_vars["awaiting_major_title"] = False
            set_major(f"{current_major} {clean_heading(line).strip()}")
        else:
            append_text(line, page_num)

    def handle_table_safe(md: str, page_num: int):
        nonlocal circle_title, circle_texts, circle_tables, circle_page, pending_table_idx
        md, inferred_heading = trim_table_markdown(md)
        if not md or is_noise_table(md):
            return
        md = prepend_table_header(md, pending_table_header_lines)
        pending_table_header_lines.clear()
        if inferred_heading:
            inferred_heading = clean_heading(inferred_heading)
            m_snt = SUB_NUM_TITLE_RE.match(inferred_heading)
            m_c = CIRCLE_RE.match(inferred_heading)
            if m_snt and (current_sub_num != m_snt.group(1) or current_sub_title != m_snt.group(2)):
                start_sub(m_snt.group(1), m_snt.group(2), page_num)
            elif m_c and in_section_9 and circle_title != m_c.group(1).strip():
                flush_circle()
                circle_title = m_c.group(1).strip()
                circle_texts = []
                circle_tables = []
                circle_page = page_num
        if in_section_9 and circle_title:
            circle_tables.append(md)
            pending_table_idx = len(circle_tables) - 1
        else:
            current_tables.append(md)
            pending_table_idx = len(current_tables) - 1

    with pdfplumber.open(pdf_path) as pdf:
        for page_num in SECTION_1_PAGES:
            if page_num in table_pages and page_num in upstage_by_page:
                for el in upstage_by_page[page_num]:
                    category = el.get("category")
                    if category == "footer":
                        continue
                    raw = element_text(el)
                    if category == "table":
                        handle_table_safe(el["content"].get("markdown", ""), page_num)
                    else:
                        if is_footer_text(raw):
                            continue
                        for line in split_element_lines(raw):
                            handle_text_line(line, page_num)
            else:
                page = pdf.pages[page_num - 1]
                for line in (page.extract_text() or "").splitlines():
                    handle_text_line(line, page_num)

    flush_current()
    return chunks


# ── Section II parsing (pages 17–38) ──────────────────────
def parse_section_2(pdf_path: Path) -> list[dict]:
    chunks = []
    current_qa_section = ""
    current_q_num  = None
    current_q_text = []
    current_a_text = []
    current_page   = None
    state          = None  # "Q" or "A"
    used_table_pages = set()

    def flush():
        nonlocal used_table_pages
        if current_q_num is not None and current_a_text:
            tables = []
            if current_page is not None and current_page not in used_table_pages:
                tables = section_2_tables_by_page.get(current_page, [])
                if tables:
                    used_table_pages.add(current_page)
            chunks.append({
                "source": "2026",
                "section": "Ⅱ. 자주하는 질문",
                "parent_subject": current_qa_section,
                "subject": ' '.join(current_q_text).strip(),
                "text": clean_chunk_text(' '.join(current_a_text)),
                "tables": tables,
                "page": current_page,
            })

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num not in SECTION_2_PAGES:
                continue

            raw = page.extract_text() or ""
            for line in raw.splitlines():
                line = line.strip()
                if not line or is_noise(line):
                    continue

                # Sub-category heading "01 의료급여기준" etc.
                if MAJOR_HEADING_RE.match(line):
                    flush()
                    current_qa_section = line
                    current_q_num = None
                    state = None
                    continue

                # Q number: "Q1", "Q2", ...
                m_q = Q_RE.match(line)
                if m_q:
                    flush()
                    current_q_num  = m_q.group(1)
                    current_q_text = []
                    current_a_text = []
                    current_page   = page_num
                    state          = "Q"
                    continue

                # Answer line: "A ..."
                if A_RE.match(line):
                    state = "A"
                    current_a_text.append(line[2:].strip())
                    continue

                if state == "Q":
                    current_q_text.append(line)
                elif state == "A":
                    current_a_text.append(line)

    flush()
    return chunks


# ── Run & save ─────────────────────────────────────────────
sec1_chunks = parse_section_1_hybrid(PDF_PATH, TABLE_PAGES)
sec2_chunks = parse_section_2(PDF_PATH)
all_chunks  = sec1_chunks + sec2_chunks

print(f"섹션 Ⅰ: {len(sec1_chunks)}개")
print(f"섹션 Ⅱ: {len(sec2_chunks)}개")
print(f"전체: {len(all_chunks)}개")

output_path = BASE_DIR.parent / "data" / "chunks" / "all_chunks(2026).json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(all_chunks, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"저장 완료: {output_path}")
