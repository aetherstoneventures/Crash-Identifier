"""
Complete Model Training Pipeline

This script:
1. Loads raw data from database
2. Performs feature engineering (28 indicators)
3. Trains all 7 models (5 crash + 2 bottom)
4. Saves trained models to data/models/
5. Saves processed data to data/processed/
"""

import logging
import sys
import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import RANDOM_STATE
from src.utils.database import DatabaseManager, Indicator, Prediction
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.models.crash_prediction.crash_labeler import CrashLabeler
from src.models.bottom_prediction.bottom_labeler import BottomLabeler
from src.models.crash_prediction import (
    SVMCrashModel, RandomForestCrashModel, GradientBoostingCrashModel,
    NeuralNetworkCrashModel, EnsembleCrashModel
)
from src.models.crash_prediction.advanced_ensemble_model import AdvancedEnsembleModel
from src.models.crash_prediction.advanced_statistical_model import AdvancedStatisticalModel
from src.models.bottom_prediction import MLPBottomModel, LSTMBottomModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    """Load raw data from database."""
    logger.info("Loading raw data from database...")
    db_manager = DatabaseManager()
    session = db_manager.get_session()

    try:
        indicators = session.query(Indicator).all()

        if not indicators:
            raise ValueError("No data found in database. Run backfill_data.py first.")

        data = pd.DataFrame([
            {
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
                'shiller_pe': ind.shiller_pe,
                'margin_debt': ind.margin_debt,
                'put_call_ratio': ind.put_call_ratio,
            }
            for ind in indicators
        ])

        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)

        logger.info(f"Loaded {len(data)} records from database")
        return data
    finally:
        session.close()


def save_raw_data(raw_data: pd.DataFrame) -> None:
    """Save raw data to data/raw/."""
    os.makedirs('data/raw', exist_ok=True)
    filepath = 'data/raw/raw_data.csv'
    raw_data.to_csv(filepath, index=False)
    logger.info(f"Saved raw data to {filepath}")


def save_processed_data(processed_data: pd.DataFrame) -> None:
    """Save processed data to data/processed/."""
    os.makedirs('data/processed', exist_ok=True)
    filepath = 'data/processed/features.csv'
    processed_data.to_csv(filepath, index=False)
    logger.info(f"Saved processed data to {filepath}")


def save_model(model, model_name: str) -> None:
    """Save trained model to data/models/."""
    os.makedirs('data/models', exist_ok=True)
    filepath = f'data/models/{model_name}.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved {model_name} to {filepath}")


def train_crash_models(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Train all crash prediction models."""
    logger.info("=" * 80)
    logger.info("Training Crash Prediction Models")
    logger.info("=" * 80)
    
    models = {}
    
    # SVM Model
    logger.info("Training SVM Model...")
    svm_model = SVMCrashModel()
    svm_model.train(X_train, y_train, X_val, y_val)
    save_model(svm_model, 'svm_crash_model')
    models['svm'] = svm_model

    # Random Forest Model
    logger.info("Training Random Forest Model...")
    rf_model = RandomForestCrashModel()
    rf_model.train(X_train, y_train, X_val, y_val)
    save_model(rf_model, 'rf_crash_model')
    models['rf'] = rf_model

    # Gradient Boosting Model
    logger.info("Training Gradient Boosting Model...")
    gb_model = GradientBoostingCrashModel()
    gb_model.train(X_train, y_train, X_val, y_val)
    save_model(gb_model, 'gb_crash_model')
    models['gb'] = gb_model

    # Neural Network Model
    logger.info("Training Neural Network Model...")
    nn_model = NeuralNetworkCrashModel()
    nn_model.train(X_train, y_train, X_val, y_val)
    save_model(nn_model, 'nn_crash_model')
    models['nn'] = nn_model

    # Ensemble Model
    logger.info("Training Ensemble Model...")
    ensemble = EnsembleCrashModel()
    ensemble.add_model('svm', svm_model)
    ensemble.add_model('rf', rf_model)
    ensemble.add_model('gb', gb_model)
    ensemble.add_model('nn', nn_model)
    ensemble.calculate_optimal_weights(X_val, y_val)
    save_model(ensemble, 'ensemble_crash_model')
    models['ensemble'] = ensemble

    logger.info("✓ All crash models trained successfully")
    return models


def train_advanced_models(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         feature_names: list) -> dict:
    """Train advanced crash prediction models with enhanced techniques."""
    logger.info("=" * 80)
    logger.info("Training Advanced Crash Prediction Models")
    logger.info("=" * 80)

    models = {}

    # Advanced Ensemble Model (with SMOTE, feature engineering, stacking)
    logger.info("Training Advanced Ensemble Model...")
    try:
        adv_ensemble = AdvancedEnsembleModel()
        adv_ensemble.train(X_train, y_train, feature_names)
        save_model(adv_ensemble, 'advanced_ensemble_crash_model')
        models['advanced_ensemble'] = adv_ensemble
        logger.info("✓ Advanced Ensemble Model trained successfully")
    except Exception as e:
        logger.warning(f"⚠ Advanced Ensemble training failed: {e}")

    # Advanced Statistical Model (with dynamic thresholds, adaptive weights)
    logger.info("Training Advanced Statistical Model...")
    try:
        adv_stat = AdvancedStatisticalModel()
        adv_stat.fit(X_train, y_train)  # Use fit() instead of train()
        save_model(adv_stat, 'advanced_statistical_crash_model')
        models['advanced_statistical'] = adv_stat
        logger.info("✓ Advanced Statistical Model trained successfully")
    except Exception as e:
        logger.warning(f"⚠ Advanced Statistical training failed: {e}")

    logger.info("✓ Advanced models training complete")
    return models


def train_bottom_models(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Train all bottom prediction models."""
    logger.info("=" * 80)
    logger.info("Training Bottom Prediction Models")
    logger.info("=" * 80)
    
    models = {}
    
    # MLP Bottom Model
    logger.info("Training MLP Bottom Model...")
    mlp_model = MLPBottomModel()
    mlp_model.train(X_train, y_train, X_val, y_val)
    save_model(mlp_model, 'mlp_bottom_model')
    models['mlp'] = mlp_model
    
    # LSTM Bottom Model
    logger.info("Training LSTM Bottom Model...")
    lstm_model = LSTMBottomModel()
    lstm_model.train(X_train, y_train, X_val, y_val)
    save_model(lstm_model, 'lstm_bottom_model')
    models['lstm'] = lstm_model
    
    logger.info("✓ All bottom models trained successfully")
    return models


def store_calculated_indicators(raw_data, processed_data):
    """Store calculated 28 indicators in the database."""
    logger.info("Storing calculated indicators in database...")

    try:
        db_manager = DatabaseManager()
        session = db_manager.get_session()

        # Get the 28 calculated indicators from processed data
        # processed_data contains normalized features, we need raw calculated indicators
        from src.feature_engineering.crash_indicators import CrashIndicators
        calculated_indicators = CrashIndicators.calculate_all_indicators(raw_data)

        # Merge with raw data to get dates
        raw_data_copy = raw_data.copy()
        raw_data_copy['date'] = pd.to_datetime(raw_data_copy['date']).dt.date

        for idx, row in raw_data_copy.iterrows():
            try:
                date = row['date']

                # Get existing indicator or create new
                existing = session.query(Indicator).filter_by(date=date).first()
                if not existing:
                    continue  # Skip if raw indicator doesn't exist

                # Update with calculated indicators
                if idx < len(calculated_indicators):
                    calc_row = calculated_indicators.iloc[idx]
                    existing.yield_spread_10y_3m = float(calc_row.get('yield_spread_10y_3m', 0))
                    existing.yield_spread_10y_2y = float(calc_row.get('yield_spread_10y_2y', 0))
                    existing.vix_level = float(calc_row.get('vix_level', 0))
                    existing.vix_change_rate = float(calc_row.get('vix_change_rate', 0))
                    existing.realized_volatility = float(calc_row.get('realized_volatility', 0))
                    existing.sp500_momentum_200d = float(calc_row.get('sp500_momentum_200d', 0))
                    existing.sp500_drawdown = float(calc_row.get('sp500_drawdown', 0))
                    existing.debt_service_ratio = float(calc_row.get('debt_service_ratio', 0))
                    existing.credit_gap = float(calc_row.get('credit_gap', 0))
                    existing.corporate_debt_growth = float(calc_row.get('corporate_debt_growth', 0))
                    existing.household_debt_growth = float(calc_row.get('household_debt_growth', 0))
                    existing.m2_growth = float(calc_row.get('m2_growth', 0))
                    existing.buffett_indicator = float(calc_row.get('buffett_indicator', 0))
                    existing.sp500_pb_ratio = float(calc_row.get('sp500_pb_ratio', 0))
                    existing.earnings_yield_spread = float(calc_row.get('earnings_yield_spread', 0))
                    existing.margin_debt_growth = float(calc_row.get('margin_debt_growth', 0))
                    existing.market_breadth = float(calc_row.get('market_breadth', 0))
                    existing.sahm_rule = float(calc_row.get('sahm_rule', 0))
                    existing.gdp_growth = float(calc_row.get('gdp_growth', 0))
                    existing.industrial_production_growth = float(calc_row.get('industrial_production_growth', 0))
                    existing.housing_starts_growth = float(calc_row.get('housing_starts_growth', 0))

                    session.commit()
            except Exception as e:
                logger.warning(f"Failed to update indicators for {date}: {e}")
                continue

        session.close()
        logger.info("✓ Calculated indicators stored in database")

    except Exception as e:
        logger.error(f"Failed to store calculated indicators: {e}", exc_info=True)
        if session:
            session.close()


def generate_and_store_predictions(raw_data, processed_data, crash_models, bottom_models):
    """Generate predictions and store them in the database."""
    logger.info("Generating predictions for all data points...")

    try:
        db_manager = DatabaseManager()
        session = db_manager.get_session()

        # Get models - prefer advanced models if available
        adv_ensemble = crash_models.get('advanced_ensemble')
        adv_stat = crash_models.get('advanced_statistical')
        rf_model = crash_models.get('rf')
        gb_model = crash_models.get('gb')
        mlp_model = bottom_models.get('mlp')

        # Generate predictions for each date
        predictions_to_store = []

        for idx in range(len(processed_data)):
            X_sample = processed_data.iloc[idx:idx+1].values

            # Get the date from raw_data - ensure it's a proper date object
            try:
                if 'date' in raw_data.columns:
                    pred_date_val = raw_data['date'].iloc[idx]
                else:
                    pred_date_val = raw_data.index[idx]

                # Convert to date object
                if isinstance(pred_date_val, pd.Timestamp):
                    pred_date = pred_date_val.date()
                elif isinstance(pred_date_val, str):
                    pred_date = pd.to_datetime(pred_date_val).date()
                else:
                    pred_date = pred_date_val
            except Exception as e:
                logger.warning(f"Could not parse date at index {idx}: {e}")
                continue

            # Get crash probability from best available models
            crash_probs = []

            # Try advanced ensemble first (best performance)
            if adv_ensemble:
                try:
                    adv_prob = adv_ensemble.predict_proba(X_sample)[0, 1]
                    crash_probs.append(adv_prob * 1.5)  # Weight advanced model higher
                    if idx < 5:
                        logger.debug(f"Advanced Ensemble prediction at idx {idx}: {adv_prob}")
                except Exception as e:
                    logger.debug(f"Advanced Ensemble prediction failed at idx {idx}: {e}")

            # Try advanced statistical model
            if adv_stat:
                try:
                    stat_proba_result = adv_stat.predict_proba(X_sample)
                    # Statistical model returns 1D array of probabilities
                    if isinstance(stat_proba_result, np.ndarray):
                        if stat_proba_result.ndim == 1:
                            stat_prob = float(stat_proba_result[0])
                        else:
                            stat_prob = float(stat_proba_result[0, 1])
                    else:
                        stat_prob = float(stat_proba_result)
                    crash_probs.append(stat_prob * 1.2)  # Weight statistical model
                    if idx < 5:
                        logger.debug(f"Advanced Statistical prediction at idx {idx}: {stat_prob}")
                except Exception as e:
                    logger.debug(f"Advanced Statistical prediction failed at idx {idx}: {e}")

            # Fallback to RF and GB if advanced models not available
            if rf_model and len(crash_probs) < 2:
                try:
                    rf_proba_result = rf_model.predict_proba(X_sample)
                    if isinstance(rf_proba_result, np.ndarray):
                        if rf_proba_result.ndim == 1:
                            rf_prob = float(rf_proba_result[0])
                        else:
                            rf_prob = float(rf_proba_result[0, 1])
                    else:
                        rf_prob = float(rf_proba_result)
                    crash_probs.append(rf_prob)
                    if idx < 5:
                        logger.debug(f"RF prediction at idx {idx}: {rf_prob}")
                except Exception as e:
                    logger.debug(f"RF prediction failed at idx {idx}: {e}")

            if gb_model and len(crash_probs) < 2:
                try:
                    gb_proba_result = gb_model.predict_proba(X_sample)
                    if isinstance(gb_proba_result, np.ndarray):
                        if gb_proba_result.ndim == 1:
                            gb_prob = float(gb_proba_result[0])
                        else:
                            gb_prob = float(gb_proba_result[0, 1])
                    else:
                        gb_prob = float(gb_proba_result)
                    crash_probs.append(gb_prob)
                    if idx < 5:
                        logger.debug(f"GB prediction at idx {idx}: {gb_prob}")
                except Exception as e:
                    logger.debug(f"GB prediction failed at idx {idx}: {e}")

            # Average the probabilities and normalize to 0-1 range
            if crash_probs:
                crash_prob = np.mean(crash_probs)
                crash_prob = np.clip(crash_prob, 0, 1)  # Ensure in valid range
                if idx < 5:
                    logger.debug(f"Final crash probability at idx {idx}: {crash_prob}")
            else:
                logger.warning(f"No valid predictions at idx {idx}, using default 0.5")
                crash_prob = 0.5

            # Confidence interval - calculate based on ensemble disagreement
            if len(crash_probs) > 1:
                # Use standard deviation of predictions as uncertainty measure
                conf_std = np.std(crash_probs)
                conf_lower = max(0, crash_prob - 1.96 * conf_std)
                conf_upper = min(1, crash_prob + 1.96 * conf_std)
            else:
                # Fallback: use fixed offset if only one model available
                conf_lower = max(0, crash_prob - 0.15)
                conf_upper = min(1, crash_prob + 0.15)

            # Get bottom prediction
            bottom_date_final = None
            recovery_date_final = None

            if mlp_model:
                try:
                    days_to_bottom = mlp_model.predict(X_sample)[0]
                    if days_to_bottom > 0 and days_to_bottom < 365:
                        bottom_date_ts = pd.Timestamp(pred_date) + pd.Timedelta(days=int(days_to_bottom))
                        bottom_date_final = bottom_date_ts.date()
                        recovery_date_ts = bottom_date_ts + pd.Timedelta(days=60)
                        recovery_date_final = recovery_date_ts.date()
                except:
                    pass

            # Create prediction record
            prediction = Prediction(
                prediction_date=pred_date,
                crash_probability=float(crash_prob),
                bottom_prediction_date=bottom_date_final,
                recovery_prediction_date=recovery_date_final,
                confidence_interval_lower=float(conf_lower),
                confidence_interval_upper=float(conf_upper),
                model_version="v1.0"
            )
            predictions_to_store.append(prediction)

        # Clear old predictions and store new ones
        session.query(Prediction).delete()
        session.add_all(predictions_to_store)
        session.commit()

        logger.info(f"✓ Stored {len(predictions_to_store)} predictions in database")
        session.close()

    except Exception as e:
        logger.error(f"Failed to generate/store predictions: {e}", exc_info=True)
        if session:
            session.close()


def main():
    """Run complete model training pipeline."""
    try:
        logger.info("=" * 80)
        logger.info("Starting Complete Model Training Pipeline")
        logger.info("=" * 80)

        # Step 1: Load raw data
        raw_data = load_raw_data()
        save_raw_data(raw_data)

        # Step 2: Feature engineering
        logger.info("=" * 80)
        logger.info("Performing Feature Engineering")
        logger.info("=" * 80)
        feature_pipeline = FeaturePipeline()
        processed_data, metadata = feature_pipeline.process(raw_data)
        save_processed_data(processed_data)

        # Step 3: Prepare labels for crash prediction
        logger.info("Preparing crash prediction labels...")
        crash_labeler = CrashLabeler()
        crash_labels = crash_labeler.generate_labels(raw_data['sp500_close'])
        crash_labels = pd.Series(crash_labels, index=raw_data.index)
        crash_labels = crash_labels.fillna(0)  # Fill NaN with 0 (no crash)

        # Step 4: Prepare labels for bottom prediction
        logger.info("Preparing bottom prediction labels...")
        bottom_labeler = BottomLabeler()
        bottom_labels = bottom_labeler.generate_labels(raw_data['sp500_close'])
        bottom_labels = pd.Series(bottom_labels, index=raw_data.index)
        bottom_labels = bottom_labels.fillna(bottom_labels.median())  # Fill NaN with median

        # Step 5: Split data (80/20 train/val)
        split_idx = int(len(processed_data) * 0.8)
        X_train_crash = processed_data.iloc[:split_idx].values
        X_val_crash = processed_data.iloc[split_idx:].values
        y_train_crash = crash_labels.iloc[:split_idx].values
        y_val_crash = crash_labels.iloc[split_idx:].values

        X_train_bottom = processed_data.iloc[:split_idx].values
        X_val_bottom = processed_data.iloc[split_idx:].values
        y_train_bottom = bottom_labels.iloc[:split_idx].values
        y_val_bottom = bottom_labels.iloc[split_idx:].values

        # Step 6: Train crash models
        crash_models = train_crash_models(
            X_train_crash, y_train_crash,
            X_val_crash, y_val_crash
        )

        # Step 6.5: Train advanced crash models
        feature_names = list(processed_data.columns)
        advanced_models = train_advanced_models(
            X_train_crash, y_train_crash,
            X_val_crash, y_val_crash,
            feature_names
        )
        crash_models.update(advanced_models)

        # Step 7: Train bottom models
        bottom_models = train_bottom_models(
            X_train_bottom, y_train_bottom,
            X_val_bottom, y_val_bottom
        )

        # Step 7.5: Store calculated indicators
        logger.info("=" * 80)
        logger.info("Storing Calculated Indicators")
        logger.info("=" * 80)
        store_calculated_indicators(raw_data, processed_data)

        # Step 8: Generate and store predictions
        logger.info("=" * 80)
        logger.info("Generating Predictions")
        logger.info("=" * 80)
        generate_and_store_predictions(raw_data, processed_data, crash_models, bottom_models)

        logger.info("=" * 80)
        logger.info("✓ Model Training Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"Trained models saved to: data/models/")
        logger.info(f"Processed data saved to: data/processed/")
        logger.info(f"Predictions stored in database")

    except Exception as e:
        logger.error(f"Model training pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

