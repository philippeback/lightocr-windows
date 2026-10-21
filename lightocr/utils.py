import os
import logging

def validate_image_path(path: str) -> bool:
    """
    Check if the given path exists and is a file.
    """
    return os.path.isfile(path)

# Configure a basic logger for the package
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
