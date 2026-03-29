#!/usr/bin/env python3
"""Train advanced crash prediction models with MLflow tracking.

This script trains multiple advanced models:
- LSTM with attention mechanism
- XGBoost with Optuna optimization
- Improved statistical model
- Ensemble of all models

All models are tracked with MLflow for versioning and comparison.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import validate_api_keys, MODELS_DIR, RANDOM_STATE
from src.utils.database import DatabaseManager, Indicator, CrashEvent
from src.utils.logger import setup_logger
from src.utils.mlflow_utils import MLflowModelManager
from src.utils.walk_forward_validation import WalkForwardValidator, calculate_metrics
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.models.crash_prediction.lstm_crash_model import LSTMCrashModel
from src.models.crash_prediction.xgboost_crash_model import XGBoostCrashModel
from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3
from src.models.crash_prediction.crash_labeler import CrashLabeler

# Setup logging
logger = setup_logger(__name__)


def load_data() -> tuple:
    """Load indicators and crash events from database.
    
    Returns:
        Tuple of (indicators_df, crash_events_df)
    """
    logger.info("Loading data from database...")
    
    db_manager = DatabaseManager()
    
    with db_manager.get_session() as session:
        # Load indicators
        indicators = session.query(Indicator).order_by(Indicator.date).all()
        session.expunge_all()
        
        # Load crash events
        crash_events = session.query(CrashEvent).all()
        session.expunge_all()
    
    # Convert to DataFrames
    indicators_data = []
    for ind in indicators:
        indicators_data.append({
            'date': ind.date,
            'yield_10y_3m': ind.yield_10y_3m,
            'yield_10y_2y': ind.yield_10y_2y,
            'yield_10y': ind.yield_10y,
            'credit_spread_bbb': ind.credit_spread_bbb,
            'unemployment_rate': ind.unemployment_rate,
            'real_gdp': ind.real_gdp,
            'cpi': ind.cpi,
            'fed_funds_rate': ind.fed_funds_rate,
            'industrial_production': ind.industrial_production,
            'sp500_close': ind.sp500_close,
            'sp500_volume': ind.sp500_volume,
            'vix_close': ind.vix_close,
            'consumer_sentiment': ind.consumer_sentiment,
            'housing_starts': ind.housing_starts,
            'm2_money_supply': ind.m2_money_supply,
            'debt_to_gdp': ind.debt_to_gdp,
            'savings_rate': ind.savings_rate,
            'lei': ind.lei,
            'margin_debt': ind.margin_debt,
            'put_call_ratio': ind.put_call_ratio
        })
    
    indicators_df = pd.DataFrame(indicators_data)
    
    crash_events_data = []
    for event in crash_events:
        crash_events_data.append({
            'start_date': event.start_date,
            'end_date': event.end_date,
            'trough_date': event.trough_date,
            'max_drawdown': event.max_drawdown
        })
    
    crash_events_df = pd.DataFrame(crash_events_data)
    
    logger.info(f"Loaded {len(indicators_df)} indicators and {len(crash_events_df)} crash events")
    
    return indicators_df, crash_events_df


def create_labels(indicators_df: pd.DataFrame, crash_events_df: pd.DataFrame, lookforward: int = 60) -> pd.Series:
    """Create crash labels using CrashLabeler (rolling-peak drawdown method).
    
    Args:
        indicators_df: DataFrame with indicators (must have sp500_close column)
        crash_events_df: DataFrame with crash events (unused — labels derived from price)
        lookforward: Days to look forward for crash prediction
        
    Returns:
        Series with crash labels (0 or 1)
    """
    logger.info(f"Creating crash labels with {lookforward}-day lookforward...")
    
    labeler = CrashLabeler(lookforward_window=lookforward)
    labels = labeler.label(indicators_df['sp500_close'])
    
    logger.info(f"Created labels: {labels.sum()} positive samples ({labels.mean()*100:.2f}%)")
    
    return labels


def train_models():
    """Main training function."""
    logger.info("=" * 80)
    logger.info("ADVANCED MODEL TRAINING WITH MLFLOW TRACKING")
    logger.info("=" * 80)
    
    # Validate API keys
    try:
        validate_api_keys(require_fred=True)
    except ValueError as e:
        logger.error(f"API key validation failed: {e}")
        return
    
    # Initialize MLflow
    mlflow_manager = MLflowModelManager()
    
    # Load data
    indicators_df, crash_events_df = load_data()
    
    # Create features
    logger.info("Generating features...")
    feature_pipeline = FeaturePipeline()
    features_df = feature_pipeline.generate_features(indicators_df)
    
    # Create labels
    labels = create_labels(indicators_df, crash_events_df, lookforward=60)
    
    # Align features and labels
    features_df = features_df.iloc[:len(labels)]
    
    # Split data (time-series aware)
    split_idx = int(len(features_df) * 0.8)
    X_train = features_df.iloc[:split_idx]
    y_train = labels.iloc[:split_idx]
    X_val = features_df.iloc[split_idx:]
    y_val = labels.iloc[split_idx:]
    
    logger.info(f"Train set: {len(X_train)} samples ({y_train.sum()} crashes)")
    logger.info(f"Val set: {len(X_val)} samples ({y_val.sum()} crashes)")
    
    # ========================================================================
    # 1. TRAIN LSTM MODEL
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Training LSTM Model")
    logger.info("=" * 80)
    
    lstm_model = LSTMCrashModel(
        sequence_length=60,
        units=128,
        num_layers=3,
        dropout=0.3,
        use_attention=True
    )
    
    lstm_results = lstm_model.train(X_train, y_train, X_val, y_val)
    
    # Log to MLflow
    mlflow_manager.log_model(
        model=lstm_model,
        model_name="crash_predictor_lstm",
        model_type="keras",
        metrics=lstm_results['metrics'],
        params={
            'sequence_length': lstm_model.sequence_length,
            'units': lstm_model.units,
            'num_layers': lstm_model.num_layers,
            'dropout': lstm_model.dropout,
            'use_attention': lstm_model.use_attention
        },
        artifacts={'scaler': lstm_model.scaler},
        register=True
    )
    
    logger.info(f"LSTM Model Metrics: {lstm_results['metrics']}")
    
    # ========================================================================
    # 2. TRAIN XGBOOST MODEL
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Training XGBoost Model with Optuna Optimization")
    logger.info("=" * 80)
    
    xgb_model = XGBoostCrashModel(
        optimize_hyperparams=True,
        n_trials=50,  # Reduced for faster training
        timeout=1800  # 30 minutes
    )
    
    xgb_results = xgb_model.train(X_train, y_train, X_val, y_val)
    
    # Log to MLflow
    mlflow_manager.log_model(
        model=xgb_model,
        model_name="crash_predictor_xgboost",
        model_type="sklearn",
        metrics=xgb_results['metrics'],
        params=xgb_results['best_params'],
        artifacts={
            'feature_importance': xgb_model.get_feature_importance(),
            'shap_values': xgb_model.get_shap_values()
        },
        register=True
    )
    
    logger.info(f"XGBoost Model Metrics: {xgb_results['metrics']}")
    
    # ========================================================================
    # 3. TRAIN IMPROVED STATISTICAL MODEL
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Training Statistical Model V3")
    logger.info("=" * 80)
    
    stat_model = StatisticalModelV3()
    stat_results = stat_model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    y_val_pred = stat_model.predict_proba(X_val)
    stat_metrics = calculate_metrics(y_val.values, y_val_pred)
    
    # Log to MLflow
    mlflow_manager.log_model(
        model=stat_model,
        model_name="crash_predictor_statistical",
        model_type="custom",
        metrics=stat_metrics,
        params={'thresholds': stat_model.thresholds},
        register=True
    )
    
    logger.info(f"Statistical Model Metrics: {stat_metrics}")
    
    # ========================================================================
    # 4. COMPARE MODELS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Model Comparison")
    logger.info("=" * 80)
    
    comparison = mlflow_manager.compare_models(
        model_names=["crash_predictor_lstm", "crash_predictor_xgboost", "crash_predictor_statistical"],
        metrics=["val_auc", "val_recall", "val_precision", "val_f1"]
    )
    
    logger.info("Model Comparison:")
    for model_name, metrics in comparison.items():
        logger.info(f"\n{model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}" if value is not None else f"  {metric}: N/A")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Models saved to: {MODELS_DIR}")
    logger.info("View MLflow UI: mlflow ui --backend-store-uri data/mlflow")
    logger.info("=" * 80)


if __name__ == "__main__":
    train_models()

