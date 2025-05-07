import logging
from logging.handlers import RotatingFileHandler
import os

os.makedirs(os.path.dirname('./logs/log.txt'), exist_ok=True)
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
logger.propagate = False
if not logger.handlers:
    file_handler = RotatingFileHandler('logs/logs.txt', maxBytes=1_000_000, backupCount=1, encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter("%(levelname)s - %(message)s")
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)