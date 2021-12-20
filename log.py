import logging

_FMT_STRING = '[%(levelname)s:%(asctime)s] %(message)s'
_DATE_FMT = '%Y-%m-%d %H:%M:%S'


def setup_logging(log_file):
    logger = logging.getLogger('ucm')
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(_FMT_STRING, datefmt=_DATE_FMT))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_FMT_STRING, datefmt=_DATE_FMT))
    logger.addHandler(file_handler)


def get_logger():
    return logging.getLogger('ucm')
