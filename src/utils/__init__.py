"""Utility modules for market crash predictor."""

from src.utils.logger import setup_logger
from src.utils.config import (
    PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, LOGS_DIR, DATABASE_URL, FRED_API_KEY
)
# REMOVED DatabaseManager import to avoid mutex lock issues
# Import it directly when needed: from src.utils.database import DatabaseManager
from src.utils.validators import DataValidator

__all__ = [
    'setup_logger',
    'PROJECT_ROOT',
    'DATA_DIR',
    'RAW_DATA_DIR',
    'PROCESSED_DATA_DIR',
    'MODELS_DIR',
    'LOGS_DIR',
    'DATABASE_URL',
    'FRED_API_KEY',
    'DataValidator',
]

