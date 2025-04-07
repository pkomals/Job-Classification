import pandas as pd
import logging
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)

def is_url(path: str) -> bool:
    """
    Check if a given path is a valid URL.
    """
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def load_job_data(file_path_or_url: str) -> pd.DataFrame:
    """
    Load job description dataset from a local path or a GitHub raw URL.
    """
    try:
        if is_url(file_path_or_url):
            logging.info(f"Fetching job data from URL: {file_path_or_url}")
            df = pd.read_csv(file_path_or_url)
        else:
            logging.info(f"Loading job data from local path: {file_path_or_url}")
            df = pd.read_csv(file_path_or_url, sep=None, engine='python')

        logging.info(f"Loaded job data with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load job data: {e}")
        raise


def load_resume_data(file_path_or_url: str) -> pd.DataFrame:
    """
    Load resume dataset from a local path or a GitHub raw URL.
    """
    try:
        if is_url(file_path_or_url):
            logging.info(f"Fetching resume data from URL: {file_path_or_url}")
            df = pd.read_csv(file_path_or_url)
        else:
            logging.info(f"Loading resume data from local path: {file_path_or_url}")
            df = pd.read_csv(file_path_or_url)

        logging.info(f"Loaded resume data with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load resume data: {e}")
        raise
