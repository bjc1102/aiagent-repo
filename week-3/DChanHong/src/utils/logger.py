"""
공통 로거 유틸리티
파일 및 콘솔 출력을 지원하는 로거를 제공합니다.
"""

import logging
import sys
from pathlib import Path

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
