"""
RAG 파이프라인 실행 엔드포인트
커맨드라인에서 인덱싱 파이프라인을 실행할 수 있습니다.

사용법:
    python main.py --source <PDF 파일 경로>
    python main.py --source <디렉토리 경로> --directory
"""

# argparse: 터미널에서 명령행(command-line) 인자를 쉽게 파싱(읽기)할 수 있게 도와주는 파이썬 내장 라이브러리입니다.
import argparse

# run_indexing_pipeline: 다른 파일에서 정의한 전체 인덱싱 과정을 실행하는 함수를 가져옵니다.
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
