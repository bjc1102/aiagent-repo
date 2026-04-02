"""
RAG 파이프라인 실행 엔드포인트
커맨드라인에서 인덱싱 파이프라인을 실행할 수 있습니다.

사용법:
    python main.py --source <PDF 파일 경로>
    python main.py --source <디렉토리 경로> --directory
"""

import argparse

from src.pipeline import run_indexing_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="RAG 인덱싱 파이프라인 실행"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="PDF 파일 또는 디렉토리 경로",
    )
    parser.add_argument(
        "--directory",
        action="store_true",
        help="소스가 디렉토리인 경우 플래그",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="청크 크기 (기본값: settings에서 로드)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="청크 오버랩 크기 (기본값: settings에서 로드)",
    )

    args = parser.parse_args()

    run_indexing_pipeline(
        source=args.source,
        is_directory=args.directory,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
