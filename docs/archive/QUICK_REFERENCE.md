# Quick Reference Guide - Market Crash Predictor v2.0

**Version**: 2.0.0
**Last Updated**: November 7, 2025

---

## 🚀 Quick Start Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add FRED_API_KEY
```

### Data Collection
```bash
python scripts/collect_data.py
```

### Train Models
```bash
python scripts/training/train_advanced_models.py
```

### View MLflow
```bash
mlflow ui --backend-store-uri data/mlflow
# Open http://localhost:5000
```

### Start API
```bash
uvicorn src.api.main:app --reload
# API docs at http://localhost:8000/docs
```

### Start Dashboard
```bash
streamlit run src/dashboard/app.py
# Dashboard at http://localhost:8501
```

---

## 📁 Key Files

### Models
- `src/models/crash_prediction/lstm_crash_model.py` - LSTM with attention
- `src/models/crash_prediction/xgboost_crash_model.py` - XGBoost with Optuna
- `src/models/crash_prediction/improved_statistical_model.py` - Statistical model
- `src/models/bottom_prediction/advanced_lstm_bottom_model.py` - Bottom predictor

### Utilities
- `src/utils/mlflow_utils.py` - MLflow integration
- `src/utils/monitoring.py` - Prometheus metrics
- `src/utils/backup.py` - Database backups
- `src/utils/walk_forward_validation.py` - Time-series CV
- `src/utils/config.py` - Configuration
- `src/utils/database.py` - Database ORM

### Scripts
- `scripts/training/train_advanced_models.py` - Train all models
- `scripts/database/migrate_to_postgresql.py` - PostgreSQL migration
- `scripts/collect_data.py` - Data collection

### API
- `src/api/main.py` - FastAPI REST API

### Documentation
- `README.md` - Main documentation
- `CHANGELOG.md` - Version history
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `QUICK_REFERENCE.md` - This file

---

## 🔧 Configuration (.env)

### Required
```bash
FRED_API_KEY=your_fred_api_key_here
```

### Optional (Real Data)
```bash
CBOE_API_KEY=your_cboe_api_key_here      # Real Put/Call Ratio
FINRA_API_KEY=your_finra_api_key_here    # Real Margin Debt
```

### Database
```bash
# SQLite (default)
DATABASE_URL=sqlite:///data/market_crash.db

# PostgreSQL (production)
DATABASE_URL=postgresql://user:password@localhost:5432/crash_predictor
POSTGRESQL_URL=postgresql://user:password@localhost:5432/crash_predictor
```

### MLflow
```bash
MLFLOW_TRACKING_URI=data/mlflow
MLFLOW_REGISTRY_URI=data/mlflow
```

### API
```bash
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
```

### Model Parameters
```bash
LSTM_UNITS=128
LSTM_LAYERS=3
LSTM_DROPOUT=0.3
LSTM_SEQUENCE_LENGTH=60
LSTM_BATCH_SIZE=32
LSTM_EPOCHS=100
LSTM_LEARNING_RATE=0.001

OPTUNA_N_TRIALS=100
OPTUNA_TIMEOUT=3600

WALK_FORWARD_WINDOW_SIZE=252
WALK_FORWARD_STEP_SIZE=21
WALK_FORWARD_MIN_TRAIN_SIZE=1260
```

### Monitoring & Backup
```bash
ENABLE_MONITORING=true
ALERT_THRESHOLD=0.7

BACKUP_ENABLED=true
BACKUP_INTERVAL_HOURS=24
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESS=true
```

---

## 📊 API Endpoints

### Health
```bash
GET /health
```

### Predictions
```bash
# Latest prediction
GET /predictions/latest?model_name=crash_predictor_xgboost

# Historical predictions
GET /predictions/historical?start_date=2024-01-01&end_date=2024-12-31&limit=100
```

### Models
```bash
# List models
GET /models/list

# Model metrics
GET /models/metrics/crash_predictor_xgboost

# Compare models
GET /models/compare?models=crash_predictor_lstm&models=crash_predictor_xgboost
```

### Crashes
```bash
# Historical crashes
GET /crashes/historical
```

---

## 🗄️ Database Operations

### Backup
```bash
# Create backup
python -c "from src.utils.backup import DatabaseBackup; DatabaseBackup().create_backup()"

# List backups
python -c "from src.utils.backup import DatabaseBackup; print(DatabaseBackup().list_backups())"

# Restore backup
python -c "from src.utils.backup import DatabaseBackup; DatabaseBackup().restore_backup('path/to/backup.db.gz')"
```

### PostgreSQL Migration
```bash
# 1. Create database
createdb crash_predictor

# 2. Update .env
POSTGRESQL_URL=postgresql://user:password@localhost:5432/crash_predictor

# 3. Run migration
python scripts/database/migrate_to_postgresql.py

# 4. Update DATABASE_URL
DATABASE_URL=postgresql://user:password@localhost:5432/crash_predictor
```

---

## 📈 MLflow Operations

### View Experiments
```bash
mlflow ui --backend-store-uri data/mlflow
# Open http://localhost:5000
```

### Compare Models (Python)
```python
from src.utils.mlflow_utils import MLflowModelManager

mgr = MLflowModelManager()
comparison = mgr.compare_models(
    model_names=['crash_predictor_lstm', 'crash_predictor_xgboost'],
    metrics=['val_auc', 'val_recall', 'val_precision']
)
print(comparison)
```

### Promote Model
```python
from src.utils.mlflow_utils import MLflowModelManager

mgr = MLflowModelManager()
mgr.promote_model('crash_predictor_xgboost', stage='Production')
```

### Get Best Model
```python
from src.utils.mlflow_utils import MLflowModelManager

mgr = MLflowModelManager()
best_run = mgr.get_best_model('crash_predictor_xgboost', metric='val_auc')
print(f"Best AUC: {best_run['metrics.val_auc']}")
```

---

## 🔍 Monitoring

### Prometheus Metrics
```bash
# Start API
uvicorn src.api.main:app

# View metrics
curl http://localhost:8000/metrics
```

### Key Metrics
- `model_auc{model_name="...", dataset="..."}` - Model AUC
- `crash_probability{model_name="..."}` - Current crash probability
- `predictions_total{model_name="..."}` - Total predictions
- `api_requests_total{endpoint="...", method="...", status="..."}` - API requests
- `model_training_total{model_name="...", status="..."}` - Training runs

---

## 🧪 Testing

### Run All Tests
```bash
pytest
```

### Run Specific Tests
```bash
pytest tests/test_models/
pytest tests/test_data_collection/
pytest tests/test_integration/
```

### With Coverage
```bash
pytest --cov=src --cov-report=html
```

---

## 🐛 Troubleshooting

### API Key Issues
```bash
# Validate API keys
python -c "from src.utils.config import validate_api_keys; validate_api_keys()"
```

### Database Issues
```bash
# Check database connection
python -c "from src.utils.database import DatabaseManager; db = DatabaseManager(); print('OK')"
```

### MLflow Issues
```bash
# Check MLflow connection
python -c "from src.utils.mlflow_utils import MLflowModelManager; mgr = MLflowModelManager(); print('OK')"
```

### Configuration Issues
```bash
# Validate configuration
python -c "from src.utils.config import validate_config; print(validate_config())"
```

---

## 📚 Documentation

- **README.md** - Main documentation with setup and usage
- **CHANGELOG.md** - Version history and migration guide
- **IMPLEMENTATION_SUMMARY.md** - Detailed implementation summary
- **docs/ARCHITECTURE.md** - System architecture
- **docs/METHODOLOGY.md** - Technical methodology
- **docs/QUICK_START_GUIDE.md** - Getting started guide

---

## 🎯 Model Comparison

| Model | Type | Strengths | Use Case |
|-------|------|-----------|----------|
| **LSTM** | Deep Learning | Sequence modeling, temporal patterns | Long-term trends |
| **XGBoost** | Gradient Boosting | Feature importance, fast inference | Feature-driven predictions |
| **Statistical** | Rule-Based | Interpretable, no training needed | Quick analysis, baseline |

---

## 💡 Tips

1. **Always backup before major changes**:
   ```bash
   python -c "from src.utils.backup import DatabaseBackup; DatabaseBackup().create_backup()"
   ```

2. **Use MLflow to track experiments**:
   - All training runs are automatically logged
   - Compare models in MLflow UI
   - Promote best models to Production

3. **Monitor system health**:
   ```bash
   curl http://localhost:8000/health
   ```

4. **Use walk-forward validation for backtesting**:
   - Prevents look-ahead bias
   - Proper time-series evaluation

5. **Check synthetic data warnings**:
   - System warns when using synthetic proxies
   - Add CBOE_API_KEY and FINRA_API_KEY for real data

---

## 🔗 Useful Links

- **FRED API**: https://fredaccount.stlouisfed.org/apikeys
- **MLflow Docs**: https://mlflow.org/docs/latest/index.html
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Prometheus Docs**: https://prometheus.io/docs/introduction/overview/

