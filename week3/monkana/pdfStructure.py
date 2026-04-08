from pathlib import Path
import json
import pdfplumber

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / Path("2024 알기 쉬운 의료급여제도.pdf")

# 사람이 보는 5, 6페이지
TARGET_PAGES = list(range(3, 16))


def inspect_tables_on_pages(pdf_path: Path, target_pages: list[int]):
    all_results = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num in target_pages:
            page = pdf.pages[page_num - 1]   # pdf.pages는 0부터 시작

            print("=" * 100)
            print(f"[PAGE {page_num}]")

            # 1) 페이지 전체 텍스트
            full_text = page.extract_text() or ""
            print("\n[PAGE TEXT PREVIEW]")
            print(full_text[:1000])

            # 2) 가장 큰 표 1개만 텍스트로
            biggest_table = page.extract_table()
            print("\n[BIGGEST TABLE]")
            print(biggest_table)

            # 3) 페이지 내 모든 표를 텍스트로
            all_tables = page.extract_tables()
            print(f"\n[ALL TABLES COUNT] {len(all_tables)}")
            for i, table in enumerate(all_tables, start=1):
                print(f"\n--- table {i} ---")
                for row in table:
                    print(row)

            # 4) 표 객체로 구조 확인
            table_objects = page.find_tables()
            print(f"\n[TABLE OBJECT COUNT] {len(table_objects)}")

            page_result = {
                "page": page_num,
                "tables": []
            }

            for t_idx, tbl in enumerate(table_objects, start=1):
                extracted = tbl.extract()   # row -> cell
                table_info = {
                    "table_index": t_idx,
                    "bbox": tbl.bbox,              # 표 전체 영역
                    "row_count": len(tbl.rows),
                    "column_count": len(tbl.columns),
                    "cell_count": len(tbl.cells),
                    "data": extracted,
                }

                print(f"\n### TABLE OBJECT {t_idx}")
                print("bbox:", tbl.bbox)
                print("rows:", len(tbl.rows))
                print("columns:", len(tbl.columns))
                print("cells:", len(tbl.cells))
                print("data:")
                for row in extracted:
                    print(row)

                page_result["tables"].append(table_info)

            all_results.append(page_result)

    return all_results


def save_results(results, output_json="pdfplumber_tables_p5_6.json"):
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {output_json}")


if __name__ == "__main__":
    results = inspect_tables_on_pages(PDF_PATH, TARGET_PAGES)
    save_results(results)