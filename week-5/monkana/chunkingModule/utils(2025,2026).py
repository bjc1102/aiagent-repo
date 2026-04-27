"""
utils(2025,2026).py
───────────────────
두 연도 청크를 합쳐 all_chunks(2025,2026).json 을 생성합니다.

동작 방식:
  1. all_chunks(2025).json 가 없으면 utils(2025).py 를 실행하여 생성
  2. all_chunks(2026).json 가 없으면 utils(2026).py 를 실행하여 생성
  3. 두 파일을 2025 → 2026 순서로 합쳐 all_chunks(2025,2026).json 저장
"""

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MONKANA_DIR = BASE_DIR.parent

CHUNKS_DIR      = MONKANA_DIR / "data" / "chunks"
OUTPUT_2025     = CHUNKS_DIR / "all_chunks(2025).json"
OUTPUT_2026     = CHUNKS_DIR / "all_chunks(2026).json"
OUTPUT_COMBINED = CHUNKS_DIR / "all_chunks(2025,2026).json"


def ensure_chunk_file(output_path: Path, script_name: str) -> None:
    if output_path.exists():
        print(f"캐시 사용: {output_path.name}")
        return
    print(f"{output_path.name} 없음 → {script_name} 실행 중...")
    subprocess.run(
        [sys.executable, str(BASE_DIR / script_name)],
        check=True,
    )
    if not output_path.exists():
        raise FileNotFoundError(f"{script_name} 실행 후에도 {output_path.name} 가 생성되지 않았습니다.")


ensure_chunk_file(OUTPUT_2025, "utils(2025).py")
ensure_chunk_file(OUTPUT_2026, "utils(2026).py")

chunks_2025 = json.loads(OUTPUT_2025.read_text(encoding="utf-8"))
chunks_2026 = json.loads(OUTPUT_2026.read_text(encoding="utf-8"))
all_chunks  = chunks_2025 + chunks_2026

print(f"2025: {len(chunks_2025)}개  |  2026: {len(chunks_2026)}개  |  전체: {len(all_chunks)}개")

OUTPUT_COMBINED.write_text(
    json.dumps(all_chunks, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"저장 완료: {OUTPUT_COMBINED}")
