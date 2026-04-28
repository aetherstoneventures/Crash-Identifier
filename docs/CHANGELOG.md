# Changelog

All notable changes to the Market Crash Predictor system.

## [3.1.0] - 2026-04-28

### Lean cleanup — v5 frozen as production

After v6, v5_multi and four future-work experiments all FAILED kill criteria,
v5 has been formally frozen as the production model and the repo trimmed to
the canonical pipeline.

**Added**
- **`run.sh`** — single-command launcher at the repo root with FRESH-vs-REUSE
  venv prompt and `--dashboard-only`/`--fresh`/`--reuse` flags.
- **`src/dashboard/pages/v5_production.py`** — fancy v5-aware dashboard page
  (live alarm state, master chart with alarm shading, BLIND scorecard, kill
  scorecard vs shelved alternatives, historical events table).
- **`🛡️ v5 Production` tab** — now the dashboard's landing page.

**Removed (transitional artifacts)**
- `scripts/run_pipeline.sh`, `scripts/run_dashboard.sh` (superseded by `run.sh`).
- `scripts/training/train_v4_walkforward_gbm.py`, `train_v6.py`,
  `train_advanced_models.py`, `scripts/training/archive/` (all obsolete).
- `scripts/data/fetch_v6_features.py`.
- `scripts/utils/phase{2..6}_*.py`, `eval_phase3.py`, `full_upgrade_pipeline.py`,
  `april2026_refresh{,_v2}.py`, `refresh_{april_24,sp500_history,v5}.py`,
  `alarm_duration_scan.py`, `analyze_results.py`, `crash_scorecard.py`,
  `critical_audit.py`, `v6_kill_or_promote.py`.
- `data/alarm_config{,_v4,_v6}.json`, `data/experiment_D_multi_asset.json`
  (round 1, superseded by `_round2`), `data/optimal_threshold.txt`,
  `data/v5_backtest.csv`, `models/v4_gbm_final.pkl`.
- `docs/archive/`, `DEEP_AUDIT_REPORT.md`.

**Pipeline order (lean)**
1. `collect_data.py` → 2. `populate_crash_events.py` →
3. `train_statistical_model_v3.py` → 4. `generate_predictions_v5.py` (StatV3
predictions) → 5. **`train_v5_walkforward.py`** (canonical v5) →
6. `train_bottom_predictor.py` → 7. `generate_bottom_predictions.py` →
8. `evaluate_crash_detection.py` → 9. dashboard.

---

## [2.0.0] - 2025-11-07

### 🎉 Major Release: Advanced ML with Production Infrastructure

This is a complete overhaul of the system with advanced machine learning models, MLflow versioning, and production-ready infrastructure.

---

### ✨ Added

#### Advanced ML Models
- **LSTM Crash Model** (`src/models/crash_prediction/lstm_crash_model.py`)
  - Bidirectional LSTM with attention mechanism
  - Sequence length: 60 days
  - 3 LSTM layers with 128 units each
  - Dropout (0.3) and L2 regularization
  - Early stopping and learning rate reduction
  - Proper anti-overfitting measures

- **XGBoost Crash Model** (`src/models/crash_prediction/xgboost_crash_model.py`)
  - Optuna hyperparameter optimization (50-100 trials)
  - SHAP values for interpretability
  - Class imbalance handling (scale_pos_weight)
  - Walk-forward validation support
  - Feature importance tracking

- **Improved Statistical Model** (`src/models/crash_prediction/improved_statistical_model.py`)
  - Multi-factor risk scoring (6 factors)
  - Dynamic threshold calibration
  - Weighted risk factors (yield curve 25%, volatility 20%, etc.)
  - Non-linear risk amplification for extreme scenarios
  - Interpretable factor scores

- **Advanced LSTM Bottom Model** (`src/models/bottom_prediction/advanced_lstm_bottom_model.py`)
  - Deep learning for market bottom prediction
  - Bidirectional LSTM with attention
  - Predicts days to bottom and recovery time
  - Proper sequence modeling

#### MLflow Integration
- **MLflow Utilities** (`src/utils/mlflow_utils.py`)
  - Complete model versioning and tracking
  - Model registry with staging (Development, Staging, Production)
  - Experiment tracking with metrics and parameters
  - Model comparison across experiments
  - Best model selection and promotion
  - Model history tracking
  - Automatic cleanup of old runs

#### Production Infrastructure
- **FastAPI REST API** (`src/api/main.py`)
  - RESTful API with automatic documentation
  - Endpoints: health, predictions, models, crashes
  - Model comparison endpoint
  - CORS middleware
  - Pydantic models for validation
  - Comprehensive error handling

- **PostgreSQL Support**
  - Migration script (`scripts/database/migrate_to_postgresql.py`)
  - Automatic table creation
  - Batch migration with validation
  - Backup before migration
  - Connection pooling configuration

- **Database Backup** (`src/utils/backup.py`)
  - Automated backup with scheduling
  - Compression support (gzip)
  - Retention policy (configurable days)
  - Restore functionality
  - Metadata tracking
  - List and cleanup old backups

- **Prometheus Monitoring** (`src/utils/monitoring.py`)
  - Metrics collection for all components
  - Decorators for automatic monitoring
  - Model performance tracking
  - API request metrics
  - Data collection metrics
  - Alert triggering
  - Health check endpoint

- **Walk-Forward Validation** (`src/utils/walk_forward_validation.py`)
  - Proper time-series cross-validation
  - Expanding window support
  - Gap between train/test to prevent leakage
  - Backtesting metrics
  - Strategy evaluation

#### Training & Scripts
- **Advanced Model Training** (`scripts/training/train_advanced_models.py`)
  - Trains all advanced models
  - MLflow tracking for all experiments
  - Model comparison and selection
  - Comprehensive logging
  - Validation metrics

### 🔧 Changed

#### Configuration
- **Enhanced config.py** (`src/utils/config.py`)
  - Added `validate_api_keys()` function with helpful error messages
  - Added `validate_config()` for comprehensive validation
  - Added CBOE_API_KEY and FINRA_API_KEY for real data
  - Added PostgreSQL configuration (POSTGRESQL_URL)
  - Added MLflow configuration
  - Added FastAPI configuration
  - Added advanced ML model parameters
  - Added monitoring and backup configuration
  - Added synthetic data warning flags
  - Auto-validation on import with warnings

- **Updated .env.example**
  - Added all new configuration options
  - Added CBOE and FINRA API keys
  - Added PostgreSQL connection string
  - Added MLflow settings
  - Added API server settings
  - Added model hyperparameters
  - Added monitoring and backup settings

#### Dependencies
- **Pinned all dependencies** in `requirements.txt`
  - Changed from `>=` to `==` for reproducibility
  - Added MLflow 2.18.0
  - Added FastAPI 0.115.5
  - Added Uvicorn 0.32.1
  - Added PostgreSQL support (psycopg2-binary 2.9.10)
  - Added Redis 5.2.0
  - Added Prometheus client 0.21.0
  - Added Optuna 4.1.0
  - Added SHAP 0.46.0
  - Added PyTorch 2.5.1
  - Added Transformers 4.46.2
  - Added LightGBM 4.5.0
  - Added CatBoost 1.2.7

#### Database
- **Fixed session management** (`src/utils/database.py`)
  - Added context manager support to `get_session()`
  - Proper session.commit() and session.rollback()
  - Automatic session.close() in finally block

- **Updated dashboard** (`src/dashboard/app.py`)
  - Fixed session leaks using context managers
  - Added session.expunge_all() before caching
  - Proper error handling

#### Documentation
- **Completely rewritten README.md**
  - Updated for v2.0 features
  - Added MLflow instructions
  - Added API documentation
  - Added PostgreSQL migration guide
  - Added monitoring section
  - Added backup instructions
  - Added model comparison table
  - Removed outdated v1.0 information

### 🗑️ Removed

#### Deleted Old Models (9 files)
- `src/models/crash_prediction/statistical_model.py` (replaced by improved version)
- `src/models/crash_prediction/advanced_statistical_model.py` (outdated)
- `src/models/crash_prediction/advanced_ml_model.py` (outdated)
- `src/models/crash_prediction/gradient_boosting_model.py` (replaced by XGBoost)
- `src/models/crash_prediction/random_forest_model.py` (outdated)
- `src/models/crash_prediction/neural_network_model.py` (replaced by LSTM)
- `src/models/crash_prediction/svm_model.py` (outdated)
- `src/models/crash_prediction/ensemble_model.py` (outdated)
- `src/models/crash_prediction/advanced_ensemble_model.py` (outdated)

#### Deleted Old Model Files (7 pickle files)
- `data/models/bottom_predictor_days_to_bottom.pkl`
- `data/models/bottom_predictor_features.pkl`
- `data/models/bottom_predictor_recovery_days.pkl`
- `data/models/gb_model_v5.pkl`
- `data/models/rf_model_v5.pkl`
- `data/models/scaler_v5.pkl`
- `data/models/statistical_model_v2.pkl`

#### Deleted Old Documentation (6 files)
- `AUDIT_CODE_LOCATIONS.md`
- `AUDIT_DETAILED_EVIDENCE.md`
- `AUDIT_REPORT.md`
- `AUDIT_SUMMARY_FOR_USER.md`
- `CLEANUP_SUMMARY.md`
- `FIXES_APPLIED.md`

### 🐛 Fixed

- **API Key Validation**: Added automatic validation with helpful error messages
- **Database Session Leaks**: Fixed using context managers
- **Synthetic Data Disclosure**: Added warnings when using synthetic proxies
- **Configuration Validation**: Auto-validation on import
- **Dependency Versions**: Pinned all versions to prevent drift

### 🔒 Security

- **API Key Protection**: Never log or expose API keys
- **Database Credentials**: Support for environment-based configuration
- **Session Management**: Proper cleanup to prevent leaks

### 📊 Performance

- **Model Optimization**: Optuna hyperparameter tuning
- **Database Pooling**: Connection pooling for PostgreSQL
- **Caching**: Redis support for caching
- **Batch Processing**: Efficient data migration

### 📝 Documentation

- **README.md**: Complete rewrite for v2.0
- **CHANGELOG.md**: This file
- **API Documentation**: Auto-generated with FastAPI
- **Code Comments**: Comprehensive docstrings

---

## [1.0.0] - 2025-10-30

### Initial Release

- Basic ML models (Gradient Boosting, Random Forest)
- SQLite database
- Streamlit dashboard
- FRED and Yahoo Finance data collection
- 81.8% recall on historical crashes
- TimeSeriesSplit validation

---

## Migration Guide: v1.0 → v2.0

### Breaking Changes

1. **Model Files**: Old pickle files are no longer compatible. Retrain via `./run.sh` (canonical: `scripts/training/train_v5_walkforward.py`)

2. **Configuration**: Update `.env` file with new variables (see `.env.example`)

3. **Database**: Optionally migrate to PostgreSQL for production use

### Migration Steps

1. **Backup your data**:
   ```bash
   python -c "from src.utils.backup import DatabaseBackup; DatabaseBackup().create_backup()"
   ```

2. **Update dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Update .env file**:
   ```bash
   cp .env.example .env.new
   # Merge your old .env with .env.new
   ```

4. **Retrain models**:
   ```bash
   ./run.sh
   ```

5. **Optional: Migrate to PostgreSQL**:
   ```bash
   python scripts/database/migrate_to_postgresql.py
   ```

6. **Test the system**:
   ```bash
   # Start API
   uvicorn src.api.main:app --reload
   
   # Start dashboard
   streamlit run src/dashboard/app.py
   
   # View MLflow
   mlflow ui --backend-store-uri data/mlflow
   ```

---

## Future Roadmap

### v2.1 (Planned)
- [ ] LightGBM crash model
- [ ] CatBoost crash model
- [ ] Transformer-based crash model
- [ ] Ensemble of all models
- [ ] Real-time data streaming
- [ ] WebSocket support for live updates

### v2.2 (Planned)
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Mobile app
- [ ] Advanced alerting (SMS, Slack, Discord)
- [ ] Trading strategy backtesting
- [ ] Portfolio optimization

### v3.0 (Future)
- [ ] Multi-asset crash prediction
- [ ] Sector-specific models
- [ ] Explainable AI dashboard
- [ ] Automated trading integration

