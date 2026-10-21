import os
import logging


def validate_image_path(path: str) -> bool:
    """
    Check if the given path exists and is a file.
    """
    return os.path.isfile(path)


# Configure a basic logger for the package without forcing root logger to INFO
logger = logging.getLogger("lightocr")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
