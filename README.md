# Market Crash Predictor - Advanced ML System v2.0

**Status**: ✅ **PRODUCTION READY** (v2.0 - Advanced ML with MLflow Versioning)
**Last Updated**: November 7, 2025

## 🎯 System Overview

An advanced machine learning system for market crash prediction using state-of-the-art deep learning and ensemble methods. The system features:

- **Advanced ML Models**:
  - LSTM with Bidirectional layers and Attention mechanism
  - XGBoost with Optuna hyperparameter optimization
  - Improved Statistical model with multi-factor risk scoring
  - All models with anti-overfitting measures

- **MLflow Integration**: Complete model versioning, tracking, and comparison
- **Production Features**: PostgreSQL support, FastAPI REST API, Prometheus monitoring, automated backups
- **Data Sources**: FRED API (16 indicators) + Yahoo Finance + FREE FINRA margin debt + FREE put/call ratio
- **Validation**: Walk-forward validation for time-series integrity

## 🆕 What's New in v2.0

### Advanced ML Models
- ✅ **LSTM with Attention**: Bidirectional LSTM with attention mechanism for interpretability
- ✅ **XGBoost with Optuna**: Automated hyperparameter optimization with SHAP values
- ✅ **Improved Statistical Model**: Multi-factor risk scoring with dynamic threshold calibration
- ✅ **Advanced Bottom Predictor**: Deep learning LSTM for market bottom prediction

### Production Infrastructure
- ✅ **MLflow Integration**: Model versioning, experiment tracking, model registry
- ✅ **PostgreSQL Support**: Production-grade database with migration scripts
- ✅ **FastAPI REST API**: RESTful API with automatic documentation
- ✅ **Prometheus Monitoring**: Metrics collection and health monitoring
- ✅ **Automated Backups**: Database backup with retention policy
- ✅ **Walk-Forward Validation**: Proper time-series cross-validation

### Code Quality
- ✅ **Deleted Old Models**: Removed 9 outdated model files
- ✅ **Consolidated Documentation**: Lean and clean directory structure
- ✅ **API Key Validation**: Automatic validation with helpful error messages
- ✅ **Session Management**: Fixed database session leaks
- ✅ **Pinned Dependencies**: All dependencies pinned to exact versions

## 📁 Directory Structure (v2.0 - Lean & Clean)

```
crash-identifier/
├── README.md                    # This file (updated for v2.0)
├── requirements.txt             # Pinned dependencies
├── .env.example                 # Environment configuration template
│
├── data/                        # Data storage
│   ├── market_crash.db          # SQLite database (or PostgreSQL)
│   ├── mlflow/                  # MLflow tracking data
│   ├── backups/                 # Database backups
│   ├── raw/                     # Raw data from APIs
│   └── logs/                    # Application logs
│
├── scripts/                     # Executable scripts
│   ├── training/
│   │   └── train_advanced_models.py  # Train all advanced models with MLflow
│   ├── database/
│   │   └── migrate_to_postgresql.py  # SQLite → PostgreSQL migration
│   ├── collect_data.py          # Data collection
│   └── run_pipeline.sh          # Main pipeline orchestrator
│
├── src/                         # Source code
│   ├── api/
│   │   └── main.py              # FastAPI REST API
│   ├── dashboard/
│   │   └── app.py               # Streamlit dashboard
│   ├── data_collection/         # Data collection modules
│   ├── feature_engineering/     # Feature pipeline
│   ├── models/
│   │   ├── crash_prediction/
│   │   │   ├── lstm_crash_model.py          # LSTM with attention
│   │   │   ├── xgboost_crash_model.py       # XGBoost with Optuna
│   │   │   └── improved_statistical_model.py # Multi-factor statistical
│   │   └── bottom_prediction/
│   │       └── advanced_lstm_bottom_model.py # Advanced bottom predictor
│   ├── utils/
│   │   ├── config.py            # Configuration with validation
│   │   ├── database.py          # SQLAlchemy ORM
│   │   ├── mlflow_utils.py      # MLflow integration
│   │   ├── monitoring.py        # Prometheus metrics
│   │   ├── backup.py            # Database backup
│   │   └── walk_forward_validation.py # Time-series CV
│   ├── alerts/                  # Alert system
│   └── scheduler/               # Task scheduling
│
├── tests/                       # Unit and integration tests
│   ├── test_data_collection/
│   ├── test_models/
│   └── test_integration/
│
├── docs/                        # Documentation
│   ├── README.md                # Documentation index
│   ├── QUICK_START_GUIDE.md     # Getting started (START HERE!)
│   ├── ARCHITECTURE.md          # System architecture
│   ├── METHODOLOGY.md           # Technical methodology
│   ├── CHANGELOG.md             # Version history
│   ├── FIXES_APPLIED.md         # Recent fixes and improvements
│   ├── IMPLEMENTATION_SUMMARY.md # Implementation details
│   └── QUICK_REFERENCE.md       # Command reference
│
└── venv/                        # Python virtual environment
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd crash-identifier

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your FRED_API_KEY (required)
# Get free key at: https://fredaccount.stlouisfed.org/apikeys
```

### 2. Collect Data
```bash
python scripts/collect_data.py
```

### 3. Train Advanced Models
```bash
python scripts/training/train_advanced_models.py
```

This will train:
- LSTM with attention mechanism
- XGBoost with Optuna optimization
- Improved statistical model

All models are tracked in MLflow.

### 4. View MLflow UI
```bash
mlflow ui --backend-store-uri data/mlflow
```

Open http://localhost:5000 to view experiments, compare models, and manage versions.

### 5. Start FastAPI Server
```bash
cd src/api
uvicorn main:app --reload
```

API documentation at: http://localhost:8000/docs

### 6. View Dashboard
```bash
streamlit run src/dashboard/app.py
```

Dashboard at: http://localhost:8501

## 📊 Model Comparison

| Model | Type | Key Features | Anti-Overfitting |
|-------|------|--------------|------------------|
| **LSTM with Attention** | Deep Learning | Bidirectional LSTM, Attention mechanism, Sequence modeling | Dropout, L2 regularization, Early stopping |
| **XGBoost with Optuna** | Gradient Boosting | Automated hyperparameter tuning, SHAP values | Walk-forward validation, Class imbalance handling |
| **Improved Statistical** | Rule-Based | Multi-factor risk scoring, Dynamic thresholds | Threshold calibration on historical data |

## 🔧 Configuration

All configuration is in `.env` file. Key settings:

```bash
# Required
FRED_API_KEY=your_fred_api_key_here

# Database (default: SQLite)
DATABASE_URL=sqlite:///data/market_crash.db
# For PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost:5432/crash_predictor

# MLflow
MLFLOW_TRACKING_URI=data/mlflow

# API
API_HOST=0.0.0.0
API_PORT=8000

# Model Parameters
LSTM_UNITS=128
LSTM_LAYERS=3
LSTM_DROPOUT=0.3
LSTM_SEQUENCE_LENGTH=60
```

## 🗄️ Database Migration (SQLite → PostgreSQL)

For production use, migrate to PostgreSQL:

```bash
# 1. Install PostgreSQL and create database
createdb crash_predictor

# 2. Update .env with PostgreSQL URL
POSTGRESQL_URL=postgresql://user:password@localhost:5432/crash_predictor

# 3. Run migration script
python scripts/database/migrate_to_postgresql.py

# 4. Update DATABASE_URL in .env
DATABASE_URL=postgresql://user:password@localhost:5432/crash_predictor
```

## 📈 Monitoring

### Prometheus Metrics

The system exposes Prometheus metrics at `/metrics` endpoint:

```bash
# Start API server
uvicorn src.api.main:app

# Metrics available at:
curl http://localhost:8000/metrics
```

Key metrics:
- `model_auc` - Model AUC scores
- `crash_probability` - Current crash probability
- `predictions_total` - Total predictions made
- `api_requests_total` - API request counts

### Health Check

```bash
curl http://localhost:8000/health
```

## 🔄 Automated Backups

Database backups are automated:

```bash
# Manual backup
python -c "from src.utils.backup import DatabaseBackup; DatabaseBackup().create_backup()"

# List backups
python -c "from src.utils.backup import DatabaseBackup; print(DatabaseBackup().list_backups())"

# Restore from backup
python -c "from src.utils.backup import DatabaseBackup; DatabaseBackup().restore_backup('path/to/backup.db.gz')"
```

Configuration in `.env`:
```bash
BACKUP_ENABLED=true
BACKUP_INTERVAL_HOURS=24
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESS=true
```
## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_models/

# Run with coverage
pytest --cov=src
```

## 📊 API Endpoints

### Health Check
```bash
GET /health
```

### Latest Prediction
```bash
GET /predictions/latest?model_name=crash_predictor_xgboost
```

### Historical Predictions
```bash
GET /predictions/historical?start_date=2024-01-01&end_date=2024-12-31&limit=100
```

### Model Comparison
```bash
GET /models/compare?models=crash_predictor_lstm&models=crash_predictor_xgboost
```

### Model Metrics
```bash
GET /models/metrics/crash_predictor_xgboost
```

Full API documentation at: http://localhost:8000/docs

## 🛡️ Anti-Overfitting Measures (v2.0)

✅ **Walk-Forward Validation**: Proper time-series cross-validation
✅ **Dropout & L2 Regularization**: In LSTM models
✅ **Early Stopping**: Prevents overtraining
✅ **Hyperparameter Optimization**: Optuna with cross-validation
✅ **Class Imbalance Handling**: scale_pos_weight in XGBoost
✅ **Attention Mechanism**: For interpretability in LSTM
✅ **SHAP Values**: Model interpretability in XGBoost

## 📈 Data Sources

| Source | Indicators | Type | Notes |
|--------|-----------|------|-------|
| **FRED** | 16 | Real | Federal Reserve Economic Data |
| **Yahoo Finance** | 2 | Real | S&P 500, VIX |
| **CBOE** | 1 | Optional | Real Put/Call Ratio (requires API key) |
| **FINRA** | 1 | Optional | Real Margin Debt (requires API key) |
| **Synthetic** | 2 | Proxy | Used when real data unavailable |

**Synthetic Data Warning**: System automatically warns when using synthetic proxies. Set `CBOE_API_KEY` and `FINRA_API_KEY` in `.env` for real data.

## 🔄 Model Versioning with MLflow

All models are versioned and tracked:

```bash
# View experiments
mlflow ui --backend-store-uri data/mlflow

# Compare models
python -c "from src.utils.mlflow_utils import MLflowModelManager; \
           mgr = MLflowModelManager(); \
           print(mgr.compare_models(['crash_predictor_lstm', 'crash_predictor_xgboost']))"

# Promote model to production
python -c "from src.utils.mlflow_utils import MLflowModelManager; \
           mgr = MLflowModelManager(); \
           mgr.promote_model('crash_predictor_xgboost', 'Production')"
```

## 📞 Support & Documentation

- **Quick Start**: `docs/QUICK_START_GUIDE.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Methodology**: `docs/METHODOLOGY.md`
- **API Docs**: http://localhost:8000/docs (when API is running)
- **MLflow UI**: http://localhost:5000 (when MLflow UI is running)

## 🎯 Key Improvements from v1.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| **Models** | Gradient Boosting + Random Forest | LSTM + XGBoost + Statistical |
| **Optimization** | Manual tuning | Optuna automated |
| **Versioning** | Pickle files | MLflow registry |
| **Database** | SQLite only | SQLite + PostgreSQL |
| **API** | None | FastAPI REST API |
| **Monitoring** | None | Prometheus metrics |
| **Backups** | Manual | Automated with retention |
| **Validation** | K-Fold | Walk-forward |
| **Interpretability** | Limited | SHAP + Attention |

---

**System Status**: ✅ Production Ready v2.0
**Last Updated**: November 7, 2025
**Models**: LSTM + XGBoost + Statistical (all with anti-overfitting)
**Infrastructure**: MLflow + FastAPI + PostgreSQL + Prometheus

