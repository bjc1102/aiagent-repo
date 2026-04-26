from week3.monkana.utils import load_and_split_pdf


def find_chunks_by_keywords(all_splits, target_keywords):
    for keyword in target_keywords:
        print("\n" + "=" * 100)
        print(f"[키워드 검색] {keyword}")
        print("=" * 100)

        matched_chunks = []

        for idx, doc in enumerate(all_splits):
            text = doc.page_content or ""
            if keyword in text:
                matched_chunks.append((idx, doc))

        print(f"포함된 청크 수: {len(matched_chunks)}")

        if not matched_chunks:
            print("해당 키워드를 포함한 청크가 없습니다.")
            continue

        for i, (chunk_idx, doc) in enumerate(matched_chunks, start=1):
            print("\n" + "-" * 100)
            print(f"[{keyword}] 청크 {i}")
            print(f"청크 인덱스: {chunk_idx}")
            print(f"메타데이터: {doc.metadata}")
            print("-" * 100)
            print(doc.page_content)


if __name__ == "__main__":
    target_keywords = [
        "65세 이상 틀니 및 치과 임플란트 본인부담률",
    ]

    all_splits = all_splits = load_and_split_pdf(
    max_characters=1000,
    combine_text_under_n_chars=300,
    new_after_n_chars=800,
)
    find_chunks_by_keywords(all_splits, target_keywords)