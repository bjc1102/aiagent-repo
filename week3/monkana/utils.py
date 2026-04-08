from pathlib import Path

from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title


BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "2024 알기 쉬운 의료급여제도.pdf"
CHROMA_DIR = BASE_DIR / "chroma_langchain_db"
GOLDEN_PATH = BASE_DIR / "goldenDataset.jsonl"


def load_and_split_pdf(
    max_characters: int = 1000,
    combine_text_under_n_chars: int = 300,
    new_after_n_chars: int = 800,
):
    elements = partition_pdf(
        filename=str(PDF_PATH),
        strategy="fast",   # 안 되면 "hi_res"도 가능
    )

    chunks = chunk_by_title(
        elements,
        max_characters=max_characters,
        combine_text_under_n_chars=combine_text_under_n_chars,
        new_after_n_chars=new_after_n_chars,
    )

    docs = []
    for i, chunk in enumerate(chunks):
        text = str(chunk).strip()
        if not text:
            continue

        metadata = {
            "chunk_id": i,
            "source": str(PDF_PATH),
        }

        # 원본 요소 메타데이터가 있으면 일부 보존
        if hasattr(chunk, "metadata") and chunk.metadata:
            if getattr(chunk.metadata, "page_number", None) is not None:
                metadata["page"] = chunk.metadata.page_number

        docs.append(Document(page_content=text, metadata=metadata))

    return docs