"""Configuration management for the market crash predictor system.

This module provides centralized configuration with validation and type safety.
All critical API keys and credentials are validated on import.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
MLFLOW_DIR = DATA_DIR / "mlflow"
LOGS_DIR = DATA_DIR / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"
BACKUPS_DIR = DATA_DIR / "backups"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
                  MLFLOW_DIR, LOGS_DIR, BACKUPS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# API CONFIGURATION WITH VALIDATION
# ============================================================================
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
# Note: CBOE and FINRA API keys are no longer needed
# - Put/Call Ratio: Calculated from SPY options via yfinance (FREE)
# - Margin Debt: Downloaded from FINRA Excel file (FREE)


def validate_api_keys(require_fred: bool = True) -> Dict[str, bool]:
    """Validate that required API keys are set.

    Args:
        require_fred: Whether FRED API key is required (default: True)

    Returns:
        Dictionary with validation status for each key

    Raises:
        ValueError: If required API keys are missing
    """
    validation_status = {
        'fred': bool(FRED_API_KEY and FRED_API_KEY != "your_fred_api_key_here"),
        'alpha_vantage': bool(ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_API_KEY != "your_alpha_vantage_key_here"),
        # CBOE and FINRA no longer needed - using FREE data sources
    }

    if require_fred and not validation_status['fred']:
        raise ValueError(
            "FRED_API_KEY is required but not set or invalid. "
            "Get your free API key at: https://fredaccount.stlouisfed.org/apikeys"
        )

    return validation_status

# ============================================================================
# DATABASE CONFIGURATION (PostgreSQL + SQLite fallback)
# ============================================================================
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DATA_DIR / 'market_crash.db'}"
)
POSTGRESQL_URL: str = os.getenv(
    "POSTGRESQL_URL",
    "postgresql://user:password@localhost:5432/crash_predictor"
)
DATABASE_POOL_SIZE: int = int(os.getenv("DATABASE_POOL_SIZE", "10"))
DATABASE_MAX_OVERFLOW: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
DATABASE_POOL_TIMEOUT: int = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))

# ============================================================================
# REDIS CONFIGURATION (for caching)
# ============================================================================
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD", None)
CACHE_TTL: int = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes default

# ============================================================================
# MLFLOW CONFIGURATION (for model versioning)
# ============================================================================
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", f"file://{MLFLOW_DIR}")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "market_crash_prediction")
MLFLOW_REGISTRY_URI: str = os.getenv("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)

# ============================================================================
# API SERVER CONFIGURATION (FastAPI)
# ============================================================================
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))
API_RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"

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

# ============================================================================
# ADVANCED ML MODEL CONFIGURATION
# ============================================================================
# LSTM/GRU Configuration
LSTM_UNITS: int = int(os.getenv("LSTM_UNITS", "128"))
LSTM_LAYERS: int = int(os.getenv("LSTM_LAYERS", "3"))
LSTM_DROPOUT: float = float(os.getenv("LSTM_DROPOUT", "0.3"))
LSTM_SEQUENCE_LENGTH: int = int(os.getenv("LSTM_SEQUENCE_LENGTH", "60"))
LSTM_BATCH_SIZE: int = int(os.getenv("LSTM_BATCH_SIZE", "32"))
LSTM_EPOCHS: int = int(os.getenv("LSTM_EPOCHS", "100"))
LSTM_EARLY_STOPPING_PATIENCE: int = int(os.getenv("LSTM_EARLY_STOPPING_PATIENCE", "15"))

# Transformer Configuration
TRANSFORMER_D_MODEL: int = int(os.getenv("TRANSFORMER_D_MODEL", "128"))
TRANSFORMER_NHEAD: int = int(os.getenv("TRANSFORMER_NHEAD", "8"))
TRANSFORMER_NUM_LAYERS: int = int(os.getenv("TRANSFORMER_NUM_LAYERS", "4"))
TRANSFORMER_DROPOUT: float = float(os.getenv("TRANSFORMER_DROPOUT", "0.2"))

# XGBoost/LightGBM/CatBoost Configuration
XGBOOST_MAX_DEPTH: int = int(os.getenv("XGBOOST_MAX_DEPTH", "6"))
XGBOOST_N_ESTIMATORS: int = int(os.getenv("XGBOOST_N_ESTIMATORS", "500"))
XGBOOST_LEARNING_RATE: float = float(os.getenv("XGBOOST_LEARNING_RATE", "0.01"))
XGBOOST_EARLY_STOPPING_ROUNDS: int = int(os.getenv("XGBOOST_EARLY_STOPPING_ROUNDS", "50"))

# Optuna Hyperparameter Optimization
OPTUNA_N_TRIALS: int = int(os.getenv("OPTUNA_N_TRIALS", "100"))
OPTUNA_TIMEOUT: int = int(os.getenv("OPTUNA_TIMEOUT", "3600"))  # 1 hour

# Walk-Forward Validation
WALK_FORWARD_WINDOW_SIZE: int = int(os.getenv("WALK_FORWARD_WINDOW_SIZE", "252"))  # 1 year
WALK_FORWARD_STEP_SIZE: int = int(os.getenv("WALK_FORWARD_STEP_SIZE", "21"))  # 1 month
WALK_FORWARD_MIN_TRAIN_SIZE: int = int(os.getenv("WALK_FORWARD_MIN_TRAIN_SIZE", "1260"))  # 5 years

# ============================================================================
# MONITORING & ALERTING CONFIGURATION
# ============================================================================
ENABLE_MONITORING: bool = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN", None)
ALERT_ON_PIPELINE_FAILURE: bool = os.getenv("ALERT_ON_PIPELINE_FAILURE", "true").lower() == "true"

# ============================================================================
# BACKUP CONFIGURATION
# ============================================================================
BACKUP_ENABLED: bool = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
BACKUP_INTERVAL_HOURS: int = int(os.getenv("BACKUP_INTERVAL_HOURS", "24"))
BACKUP_RETENTION_DAYS: int = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
BACKUP_COMPRESS: bool = os.getenv("BACKUP_COMPRESS", "true").lower() == "true"

# ============================================================================
# SYNTHETIC DATA WARNING FLAGS
# ============================================================================
USE_SYNTHETIC_PUT_CALL_RATIO: bool = os.getenv("USE_SYNTHETIC_PUT_CALL_RATIO", "true").lower() == "true"
USE_SYNTHETIC_MARGIN_DEBT: bool = os.getenv("USE_SYNTHETIC_MARGIN_DEBT", "true").lower() == "true"
WARN_SYNTHETIC_DATA: bool = os.getenv("WARN_SYNTHETIC_DATA", "true").lower() == "true"

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================
def validate_config() -> Dict[str, Any]:
    """Validate all configuration settings.

    Returns:
        Dictionary with validation results and warnings
    """
    warnings = []
    errors = []

    # Note: Put/Call Ratio and Margin Debt are now FREE
    # - Put/Call Ratio: Calculated from SPY options via yfinance
    # - Margin Debt: Downloaded from FINRA Excel file
    # No warnings needed - these are real data sources, not synthetic proxies

    # Check database configuration
    if "postgresql" in DATABASE_URL.lower():
        try:
            import psycopg2
        except ImportError:
            errors.append(
                "PostgreSQL database URL specified but psycopg2 not installed. "
                "Run: pip install psycopg2-binary"
            )

    # Check Redis configuration
    if REDIS_HOST != "localhost":
        try:
            import redis
        except ImportError:
            warnings.append(
                "Redis host specified but redis package not installed. "
                "Run: pip install redis"
            )

    return {
        'valid': len(errors) == 0,
        'warnings': warnings,
        'errors': errors,
        'synthetic_data_in_use': USE_SYNTHETIC_PUT_CALL_RATIO or USE_SYNTHETIC_MARGIN_DEBT
    }


# ============================================================================
# AUTO-VALIDATION ON IMPORT (can be disabled)
# ============================================================================
if os.getenv("SKIP_CONFIG_VALIDATION", "false").lower() != "true":
    validation_result = validate_config()

    # Print warnings
    for warning in validation_result['warnings']:
        print(f"⚠️  CONFIG WARNING: {warning}", file=sys.stderr)

    # Print errors and exit if critical
    for error in validation_result['errors']:
        print(f"❌ CONFIG ERROR: {error}", file=sys.stderr)

    if not validation_result['valid']:
        print("\n❌ Configuration validation failed. Fix errors above.", file=sys.stderr)
        # Don't exit - let the application handle it

