

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    
    Path("logs").mkdir(exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)


    file_handler = logging.FileHandler("logs/pipeline.log", mode = "a", encoding = "utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt = "%(asctime)s  %(levelname)-8s  %(name)s:%(lineno)d  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)


    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)