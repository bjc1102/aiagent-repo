"""
공통 로거 유틸리티
파일 및 콘솔 출력을 지원하는 로거를 제공합니다.
"""

# logging: 파이썬에서 프로그램의 실행 과정이나 에러 등을 기록(로그)으로 남기기 위한 내장 라이브러리입니다.
import logging
# sys: 파이썬 인터프리터가 제공하는 변수와 함수를 제어할 수 있게 해주는 모듈로, 여기서는 터미널(콘솔) 출력을 위해 사용됩니다.
import sys
# Path (pathlib 모듈): 파일 및 디렉토리 경로를 문자열이 아닌 객체로 다루어, 경로 관련 작업을 훨씬 안전하고 편리하게 해주는 라이브러리입니다.
from pathlib import Path

# LOGS_DIR: 로그 파일을 저장할 디렉토리 경로를 설정 파일에서 가져옵니다.
from src.config.settings import LOGS_DIR


def get_logger(name: str, log_file: str = "app.log") -> logging.Logger:
    """
    이름 기반 로거를 생성하여 반환합니다.
    콘솔과 파일 양쪽에 로그를 출력합니다.

    Args:
        name: 로거 이름 (보통 __name__ 사용)
        log_file: 로그 파일명 (기본값: app.log)

    Returns:
        logging.Logger: 설정된 로거 인스턴스
    """
    logger = logging.getLogger(name)

    # 이미 핸들러가 설정되어 있으면 중복 방지
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 포맷 설정
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러
    log_path = LOGS_DIR / log_file
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
