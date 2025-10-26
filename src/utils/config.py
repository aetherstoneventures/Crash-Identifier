"""Configuration management for the market crash predictor system."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# Database Configuration
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DATA_DIR / 'market_crash.db'}"
)

# Data Collection Configuration
FRED_RATE_LIMIT: int = int(os.getenv("FRED_RATE_LIMIT", "120"))
FRED_TIMEOUT: int = int(os.getenv("FRED_TIMEOUT", "30"))
YAHOO_TIMEOUT: int = int(os.getenv("YAHOO_TIMEOUT", "30"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_FACTOR: float = float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0"))

# Data Validation Configuration
EXPECTED_RANGES = {
    'yield_10y_3m': (-5.0, 5.0),
    'yield_10y_2y': (-3.0, 3.0),
    'credit_spread_bbb': (0.0, 15.0),
    'unemployment_rate': (2.0, 25.0),
    'vix_close': (5.0, 100.0),
    'sp500_close': (0.0, None),
    'consumer_sentiment': (0.0, 150.0),
}

# Scheduler Configuration
SCHEDULER_HOUR: int = int(os.getenv("SCHEDULER_HOUR", "6"))
SCHEDULER_MINUTE: int = int(os.getenv("SCHEDULER_MINUTE", "0"))

# Logging Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = str(LOGS_DIR / "market_crash_predictor.log")

# Feature Engineering Configuration
VOLATILITY_WINDOW: int = int(os.getenv("VOLATILITY_WINDOW", "20"))
MOMENTUM_WINDOW: int = int(os.getenv("MOMENTUM_WINDOW", "200"))
DRAWDOWN_WINDOW: int = int(os.getenv("DRAWDOWN_WINDOW", "252"))

# Bry-Boschan Algorithm Configuration
BOSCHAN_MIN_DURATION: int = int(os.getenv("BOSCHAN_MIN_DURATION", "15"))
BOSCHAN_MIN_AMPLITUDE: float = float(os.getenv("BOSCHAN_MIN_AMPLITUDE", "0.15"))
BOSCHAN_PHASE_LENGTH: int = int(os.getenv("BOSCHAN_PHASE_LENGTH", "6"))

# Feature Processing Configuration
FEATURE_CORRELATION_THRESHOLD: float = float(os.getenv("FEATURE_CORRELATION_THRESHOLD", "0.95"))
FEATURE_NORMALIZATION_METHOD: str = os.getenv("FEATURE_NORMALIZATION_METHOD", "standard")

# Model Configuration
RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
VALIDATION_SIZE: float = float(os.getenv("VALIDATION_SIZE", "0.1"))

# Validation thresholds
MIN_DATA_QUALITY_SCORE: float = float(os.getenv("MIN_DATA_QUALITY_SCORE", "0.95"))
MAX_MISSING_PERCENTAGE: float = float(os.getenv("MAX_MISSING_PERCENTAGE", "5.0"))

# Crash detection threshold
CRASH_DRAWDOWN_THRESHOLD: float = float(os.getenv("CRASH_DRAWDOWN_THRESHOLD", "0.20"))

# Phase 3: Crash Prediction Models Configuration
CRASH_PREDICTION_LOOKFORWARD_WINDOW: int = int(os.getenv("CRASH_PREDICTION_LOOKFORWARD_WINDOW", "60"))
CRASH_PREDICTION_AUC_TARGET: float = float(os.getenv("CRASH_PREDICTION_AUC_TARGET", "0.85"))
CRASH_PREDICTION_ACCURACY_TARGET: float = float(os.getenv("CRASH_PREDICTION_ACCURACY_TARGET", "0.85"))
CRASH_DETECTION_RECALL_TARGET: float = float(os.getenv("CRASH_DETECTION_RECALL_TARGET", "0.75"))
FALSE_ALARM_RATE_TARGET: float = float(os.getenv("FALSE_ALARM_RATE_TARGET", "0.20"))

# Phase 4: Bottom Prediction Models Configuration
BOTTOM_PREDICTION_ACCURACY_TARGET: float = float(os.getenv("BOTTOM_PREDICTION_ACCURACY_TARGET", "0.85"))
BOTTOM_PREDICTION_DAYS_TOLERANCE: int = int(os.getenv("BOTTOM_PREDICTION_DAYS_TOLERANCE", "30"))
RECOVERY_TIME_R2_TARGET: float = float(os.getenv("RECOVERY_TIME_R2_TARGET", "0.75"))

# System performance targets
DATA_COLLECTION_MAX_TIME: int = int(os.getenv("DATA_COLLECTION_MAX_TIME", "300"))
FEATURE_GENERATION_MAX_TIME: int = int(os.getenv("FEATURE_GENERATION_MAX_TIME", "120"))
MODEL_INFERENCE_MAX_TIME: int = int(os.getenv("MODEL_INFERENCE_MAX_TIME", "60"))
DASHBOARD_LOAD_MAX_TIME: int = int(os.getenv("DASHBOARD_LOAD_MAX_TIME", "2"))

