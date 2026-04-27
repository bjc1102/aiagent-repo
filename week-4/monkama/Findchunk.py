import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_all_chunks(year: str = "2025,2026") -> list[dict]:
    path = BASE_DIR / f"all_chunks({year}).json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_chunks_by_keywords(all_chunks: list[dict], target_keywords: list[str]):
    for keyword in target_keywords:
        print("\n" + "=" * 100)
        print(f"[키워드 검색] {keyword}")
        print("=" * 100)

        matched = [c for c in all_chunks if keyword in json.dumps(c, ensure_ascii=False)]

        print(f"포함된 청크 수: {len(matched)}")
        if not matched:
            print("해당 키워드를 포함한 청크가 없습니다.")
            continue

        for i, chunk in enumerate(matched, start=1):
            print("\n" + "-" * 100)
            print(f"[{keyword}] 청크 {i}")
            print(f"출처연도: {chunk.get('source')}  |  섹션: {chunk.get('section')}  |  페이지: {chunk.get('page')}")
            print(f"subject: {chunk.get('subject', '')}")
            print("-" * 100)
            print(chunk.get("text", ""))
            tables = chunk.get("tables", [])
            if tables:
                print("[표]")
                for t in tables:
                    print(t)


if __name__ == "__main__":
    user_input = input("검색할 키워드를 입력하세요 (쉼표로 구분): ").strip()
    target_keywords = [k.strip() for k in user_input.split(",") if k.strip()]

    all_chunks = load_all_chunks("2025,2026")
    find_chunks_by_keywords(all_chunks, target_keywords)
