import sys
from pathlib import Path

import ragas_eval


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "result" / "WithYearAcc"


def ensure_default_args(argv: list[str]) -> list[str]:
    args = list(argv)

    if "--output-dir" not in args:
        args.extend(["--output-dir", str(DEFAULT_OUTPUT_DIR)])

    if "--with-year-accuracy" not in args:
        args.append("--with-year-accuracy")

    if "--korean-prompts" not in args:
        args.append("--korean-prompts")

    return args


def main():
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], *ensure_default_args(sys.argv[1:])]
    ragas_eval.main()


if __name__ == "__main__":
    main()
