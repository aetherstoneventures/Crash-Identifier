"""
Hybrid Ensemble Model - Combining ML and Statistical Models
Walk-forward validation to ensure realistic performance.
"""

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sqlite3
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.crash_prediction.statistical_model_v3 import StatisticalModelV3
from src.utils.walk_forward_validation import WalkForwardValidator
from scripts.training.train_crash_detector_v6_walkforward import engineer_crash_features_v6
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def prepare_features_for_statistical(df):
    """Prepare features in the format expected by StatisticalModelV3."""
    X = pd.DataFrame(index=df.index)
    
    # Direct mappings
    X['yield_10y_2y'] = df['yield_10y_2y']
    X['yield_10y_3m'] = df['yield_10y_3m']
    X['vix_close'] = df['vix_close']
    X['credit_spread_bbb'] = df['credit_spread_bbb']
    X['unemployment_rate'] = df['unemployment_rate']
    X['industrial_production'] = df['industrial_production']
    X['consumer_sentiment'] = df['consumer_sentiment']
    
    # Calculate derived features
    X['sp500_drawdown'] = (df['sp500_close'] / df['sp500_close'].rolling(252).max() - 1)
    X['sp500_return_5d'] = df['sp500_close'].pct_change(5)
    X['sp500_return_20d'] = df['sp500_close'].pct_change(20)
    
    X['vix_change_pct'] = df['vix_close'].pct_change()
    X['vix_change_5d'] = df['vix_close'].pct_change(5)
    
    X['credit_spread_change'] = df['credit_spread_bbb'].diff(20)
    X['unemployment_change'] = df['unemployment_rate'].diff(20)
    X['industrial_production_change'] = df['industrial_production'].diff(20)
    
    # Fill NaN
    X = X.ffill().bfill().fillna(0)
    
    return X


def main():
    logger.info("=" * 80)
    logger.info("HYBRID ENSEMBLE - ML + STATISTICAL - WALK-FORWARD VALIDATION")
    logger.info("=" * 80)
    
    # Load data using sqlite3 directly
    conn = sqlite3.connect('data/market_crash.db')
    df = pd.read_sql_query("SELECT * FROM indicators ORDER BY date", conn)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    crashes = pd.read_sql_query("SELECT start_date, end_date FROM crash_events", conn)
    crash_events = list(zip(crashes['start_date'], crashes['end_date']))
    conn.close()
    
    logger.info(f"Data: {df.shape}, Crashes: {len(crash_events)}")
    
    # Prepare features for both models
    X_ml = engineer_crash_features_v6(df)
    X_stat = prepare_features_for_statistical(df)
    
    logger.info(f"ML Features: {X_ml.shape[1]}, Statistical Features: {X_stat.shape[1]}")
    
    # Create labels (90-day lookback window)
    y = pd.Series(0, index=df.index)
    for crash_start, _ in crash_events:
        crash_date = pd.to_datetime(crash_start)
        lookback = crash_date - pd.Timedelta(days=90)
        y[(df.index >= lookback) & (df.index < crash_date)] = 1
    
    logger.info(f"Crash samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    
    # Walk-forward validation
    validator = WalkForwardValidator(window_size=252, step_size=21, min_train_size=1260)
    splits = validator.expanding_window_split(X_ml, y)
    logger.info(f"Generated {len(splits)} folds\n")
    
    # Test different weighting schemes
    weight_schemes = [
        (0.5, 0.5, "50% ML + 50% Statistical"),
        (0.6, 0.4, "60% ML + 40% Statistical"),
        (0.7, 0.3, "70% ML + 30% Statistical"),
        (0.8, 0.2, "80% ML + 20% Statistical"),
    ]
    
    # Store results for each weighting scheme
    results = {}
    
    for ml_weight, stat_weight, scheme_name in weight_schemes:
        logger.info("=" * 80)
        logger.info(f"TESTING: {scheme_name}")
        logger.info("=" * 80)
        
        hybrid_metrics = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # Split data
            X_ml_train, X_ml_test = X_ml.iloc[train_idx], X_ml.iloc[test_idx]
            X_stat_train, X_stat_test = X_stat.iloc[train_idx], X_stat.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Skip invalid folds
            if y_train.sum() < 2 or y_test.sum() == 0 or y_test.sum() == len(y_test):
                continue

            # Train statistical model
            stat_model = StatisticalModelV3()
            stat_model.train(X_stat_train, y_train)
            stat_proba = stat_model.predict_proba(X_stat_test)

            # Train ML models on this fold
            scaler = StandardScaler()
            X_ml_train_scaled = scaler.fit_transform(X_ml_train)
            X_ml_test_scaled = scaler.transform(X_ml_test)

            # Apply SMOTE to training set
            try:
                smote = SMOTE(sampling_strategy=0.3, random_state=42)
                X_ml_train_balanced, y_train_balanced = smote.fit_resample(X_ml_train_scaled, y_train)
            except:
                X_ml_train_balanced = X_ml_train_scaled
                y_train_balanced = y_train

            # Train ML models
            gb = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=5,
                subsample=0.8, min_samples_split=10, min_samples_leaf=5,
                random_state=42
            )
            gb.fit(X_ml_train_balanced, y_train_balanced)

            rf = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1,
                class_weight='balanced'
            )
            rf.fit(X_ml_train_balanced, y_train_balanced)

            # Get ML predictions
            gb_proba = gb.predict_proba(X_ml_test_scaled)[:, 1]
            rf_proba = rf.predict_proba(X_ml_test_scaled)[:, 1]
            ml_proba = gb_proba * 0.7 + rf_proba * 0.3

            # Hybrid ensemble
            hybrid_proba = ml_weight * ml_proba + stat_weight * stat_proba

            # Calculate AUC
            if y_test.sum() > 0 and y_test.sum() < len(y_test):
                auc = roc_auc_score(y_test, hybrid_proba)
                hybrid_metrics.append(auc)

                if (i + 1) % 50 == 0:
                    logger.info(f"Fold {i+1}: Hybrid AUC={auc:.4f}")

        # Calculate mean performance
        if len(hybrid_metrics) > 0:
            mean_auc = np.mean(hybrid_metrics)
            std_auc = np.std(hybrid_metrics)
            results[scheme_name] = (mean_auc, std_auc)
            logger.info(f"\n{scheme_name}:")
            logger.info(f"  Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
            logger.info(f"  Valid folds: {len(hybrid_metrics)}\n")

    # Find best weighting scheme
    logger.info("\n" + "=" * 80)
    logger.info("HYBRID ENSEMBLE RESULTS SUMMARY")
    logger.info("=" * 80)

    best_scheme = None
    best_auc = 0

    for scheme_name, (mean_auc, std_auc) in results.items():
        logger.info(f"{scheme_name}: AUC {mean_auc:.4f} (+/- {std_auc:.4f})")
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_scheme = scheme_name

    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON WITH INDIVIDUAL MODELS")
    logger.info("=" * 80)
    logger.info("ML Ensemble (V6):     AUC 0.5829 (+/- 0.2922)")
    logger.info("Statistical Model V3: AUC 0.3532 (+/- 0.1624)")
    logger.info(f"BEST HYBRID:          {best_scheme} - AUC {best_auc:.4f}")

    # Performance assessment
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE ASSESSMENT")
    logger.info("=" * 80)
    if best_auc >= 0.70:
        logger.info("✅ EXCELLENT: AUC >= 0.70 (target achieved!)")
    elif best_auc >= 0.60:
        logger.info("⚠️  ACCEPTABLE: AUC >= 0.60 (needs improvement)")
    else:
        logger.info("❌ POOR: AUC < 0.60 (needs significant improvement)")
        logger.info("   → Recommendation: Proceed to Phase 4 (XGBoost/LightGBM/CatBoost)")
        logger.info("   → Or Phase 5 (Enhanced feature engineering)")

    # Train final models on all data
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING FINAL MODELS ON ALL DATA")
    logger.info("=" * 80)

    # Extract weights from best scheme
    for ml_w, stat_w, name in weight_schemes:
        if name == best_scheme:
            best_ml_weight = ml_w
            best_stat_weight = stat_w
            break

    # Train final statistical model
    stat_final = StatisticalModelV3()
    stat_final.train(X_stat, y)

    # Train final ML models
    scaler_final = StandardScaler()
    X_ml_scaled = scaler_final.fit_transform(X_ml)

    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_ml_balanced, y_balanced = smote.fit_resample(X_ml_scaled, y)
    logger.info(f"After SMOTE: {len(X_ml_balanced)} samples ({y_balanced.sum()} crashes)")

    gb_final = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=5,
        subsample=0.8, min_samples_split=10, min_samples_leaf=5,
        random_state=42
    )
    gb_final.fit(X_ml_balanced, y_balanced)

    rf_final = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1,
        class_weight='balanced'
    )
    rf_final.fit(X_ml_balanced, y_balanced)

    logger.info("✅ Final models trained on all data")

    # Save models
    os.makedirs('data/models', exist_ok=True)
    with open('data/models/gb_model_v6.pkl', 'wb') as f:
        pickle.dump(gb_final, f)
    with open('data/models/rf_model_v6.pkl', 'wb') as f:
        pickle.dump(rf_final, f)
    with open('data/models/scaler_v6.pkl', 'wb') as f:
        pickle.dump(scaler_final, f)
    with open('data/models/statistical_model_v3.pkl', 'wb') as f:
        pickle.dump(stat_final, f)

    # Save hybrid configuration
    hybrid_config = {
        'ml_weight': best_ml_weight,
        'stat_weight': best_stat_weight,
        'scheme_name': best_scheme,
        'mean_auc': best_auc,
        'std_auc': results[best_scheme][1],
        'ml_model_path': 'data/models/gb_model_v6.pkl',
        'rf_model_path': 'data/models/rf_model_v6.pkl',
        'scaler_path': 'data/models/scaler_v6.pkl',
        'stat_model_path': 'data/models/statistical_model_v3.pkl'
    }

    with open('data/models/hybrid_ensemble_config.pkl', 'wb') as f:
        pickle.dump(hybrid_config, f)

    logger.info(f"\n✅ All models saved:")
    logger.info(f"   - data/models/gb_model_v6.pkl")
    logger.info(f"   - data/models/rf_model_v6.pkl")
    logger.info(f"   - data/models/scaler_v6.pkl")
    logger.info(f"   - data/models/statistical_model_v3.pkl")
    logger.info(f"   - data/models/hybrid_ensemble_config.pkl")
    logger.info(f"   - Best scheme: {best_scheme}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()


