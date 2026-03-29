# Quick Start Guide - Market Crash Predictor v2.0

## ⚙️ Prerequisites & Setup

### 1. Environment Configuration (REQUIRED)

Before running the system, you **must** create a `.env` file with your API keys:

```bash
# Copy the example template
cp .env.example .env

# Edit the .env file and add your API keys
nano .env  # or use your preferred editor (VS Code, vim, etc.)
```

---

## 🔑 API Keys - Complete Guide

### **REQUIRED: FRED API Key** (Federal Reserve Economic Data)

**What it provides**: 16 economic indicators (unemployment, GDP, interest rates, etc.)

**How to get it (FREE - takes 2 minutes):**

1. **Visit**: https://fredaccount.stlouisfed.org/apikeys
2. **Click**: "Request API Key" (top right)
3. **Sign Up**:
   - Enter your email address
   - Create a password
   - Fill in basic information (name, organization - can be "Personal")
4. **Verify Email**: Check your inbox and click verification link
5. **Request API Key**:
   - Log in to your FRED account
   - Go to "My Account" → "API Keys"
   - Click "Request API Key"
   - Fill in the form:
     - **API Key Description**: "Market Crash Predictor" (or any name)
     - **Website URL**: Can leave blank or use "http://localhost"
   - Click "Request API key"
6. **Copy Your Key**: You'll see a 32-character key like: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`
7. **Add to .env**:
   ```bash
   FRED_API_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
   ```

**⚠️ Without this key, the system CANNOT collect data!**

---

### **Put/Call Ratio - FREE via yfinance** ✅

**What it provides**: Put/Call Ratio (market sentiment indicator)

**Good news**: The system now calculates this **automatically for FREE** using SPY options data from yfinance!

**How it works**:
- Calculates put/call ratio from SPY (S&P 500 ETF) options volume
- Uses yfinance (FREE) to get options data
- Provides total market put/call ratio across all exchanges
- More comprehensive than CBOE-only data

**No API key needed!** The system automatically calculates this from publicly available options data.

**Interpretation**:
- High ratio (>1.0) = Bearish sentiment (more puts than calls)
- Low ratio (<0.7) = Bullish sentiment (more calls than puts)

**Note**: CBOE doesn't offer a free API. Our free solution using yfinance is actually more comprehensive as it includes all exchanges, not just CBOE.

---

### **Margin Debt - FREE via FINRA** ✅

**What it provides**: Margin Debt data (market leverage indicator)

**Good news**: FINRA provides this **completely FREE** via Excel download!

**How it works**:
- FINRA publishes FREE historical margin debt data
- Available from January 1997 to present
- Updated monthly (around 3rd week of following month)
- Direct Excel file download from FINRA website

**No API key needed!** The system automatically downloads the FREE Excel file from:
https://www.finra.org/investors/learn-to-invest/advanced-investing/margin-statistics

**Data included**:
- Debit balances in customers' securities margin accounts
- Free credit balances in cash accounts
- Free credit balances in securities accounts

**Update frequency**: Monthly (published ~3 weeks after month-end)

**Note**: Unlike CBOE, FINRA actually provides FREE access to margin debt data. No subscription or API key required!

---

### **Complete .env File Example**

```bash
# ============================================================================
# REQUIRED: FRED API Key (FREE - get at https://fredaccount.stlouisfed.org/apikeys)
# ============================================================================
FRED_API_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6

# ============================================================================
# OPTIONAL: Additional Data Sources
# ============================================================================
# Alpha Vantage API Key (optional - for additional market data)
# Get at: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# NOTE: Put/Call Ratio and Margin Debt are now FREE!
# - Put/Call Ratio: Calculated automatically from SPY options (yfinance)
# - Margin Debt: Downloaded automatically from FINRA Excel file
# No additional API keys needed!

# ============================================================================
# Database Configuration
# ============================================================================
# SQLite (default - good for development)
DATABASE_URL=sqlite:///data/market_crash.db

# PostgreSQL (recommended for production)
# POSTGRESQL_URL=postgresql://user:password@localhost:5432/crash_predictor
# DATABASE_URL=postgresql://user:password@localhost:5432/crash_predictor

# ============================================================================
# MLflow Configuration
# ============================================================================
MLFLOW_TRACKING_URI=data/mlflow
MLFLOW_REGISTRY_URI=data/mlflow

# ============================================================================
# API Server Configuration
# ============================================================================
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# ============================================================================
# Model Parameters
# ============================================================================
LSTM_UNITS=128
LSTM_LAYERS=3
LSTM_DROPOUT=0.3
LSTM_SEQUENCE_LENGTH=60
LSTM_BATCH_SIZE=32
LSTM_EPOCHS=100
LSTM_LEARNING_RATE=0.001

OPTUNA_N_TRIALS=100
OPTUNA_TIMEOUT=3600

# ============================================================================
# Monitoring & Backup
# ============================================================================
ENABLE_MONITORING=true
ALERT_THRESHOLD=0.7

BACKUP_ENABLED=true
BACKUP_INTERVAL_HOURS=24
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESS=true

# ============================================================================
# Other Settings
# ============================================================================
LOG_LEVEL=INFO
RANDOM_STATE=42
```

---

### **⚠️ Important Notes**

1. **FRED API Key is REQUIRED** - The system will not work without it
2. **CBOE and FINRA are OPTIONAL** - System will use synthetic proxies if not provided
3. **Never commit .env file to git** - It's already in .gitignore
4. **Keep your API keys secret** - Don't share them publicly
5. **Free tier limits**:
   - FRED: 120 requests/minute (more than enough)
   - CBOE: Varies by subscription
   - FINRA: Varies by subscription

---

### **✅ All Data Sources Now FREE!**

**Great news**: Both Put/Call Ratio and Margin Debt are now obtained for FREE:

1. **Put/Call Ratio**: Calculated from SPY options data (yfinance) - FREE
2. **Margin Debt**: Downloaded from FINRA Excel file - FREE

**No warnings, no synthetic data, no paid subscriptions needed!**

The system will only use synthetic proxies as a fallback if the free data sources fail.

---

## 🚀 Quick Start Commands

### 1. Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy example .env file
cp .env.example .env

# Edit .env and add your FRED_API_KEY (REQUIRED)
nano .env  # or use VS Code, vim, etc.
```

### 3. Collect Data
```bash
python scripts/collect_data.py
```

### 4. Train Models
```bash
python scripts/training/train_advanced_models.py
```

### 5. Start MLflow UI (Optional)
```bash
mlflow ui --backend-store-uri data/mlflow
# Open http://localhost:5000
```

### 6. Start API Server (Optional)
```bash
uvicorn src.api.main:app --reload
# API docs at http://localhost:8000/docs
```

### 7. Start Dashboard
```bash
streamlit run src/dashboard/app.py
# Dashboard at http://localhost:8501
```

---

## 📊 System Architecture (v2.0)

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Collection                          │
│  FRED (16 indicators) + Yahoo Finance (2) + CBOE + FINRA    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Feature Engineering                         │
│         28 engineered features from 20 raw indicators        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Advanced ML Models                         │
│  LSTM (Attention) + XGBoost (Optuna) + Statistical          │
│              All tracked in MLflow                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Predictions & API                           │
│     FastAPI REST API + Streamlit Dashboard                  │
│         Prometheus Monitoring + Backups                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dashboard Overview

### Main Features
- **Current Crash Probability**: Real-time prediction from advanced models
- **Historical Predictions**: Time series of all predictions
- **20 Financial Indicators**: Interactive charts with zoom/pan
- **Model Performance**: Compare LSTM, XGBoost, and Statistical models
- **Data Quality**: Validation and coverage metrics

**Crash Probability Interpretation**:
- **0-20%**: Low risk (normal market conditions)
- **20-40%**: Moderate risk (watch for warning signs)
- **40-60%**: High risk (significant warning signs)
- **60-100%**: Very high risk (crash likely imminent)

---

## 🤖 Advanced Models (v2.0)

### Model Comparison

| Model | Type | Key Features | Anti-Overfitting |
|-------|------|--------------|------------------|
| **LSTM with Attention** | Deep Learning | Bidirectional LSTM, Attention mechanism, Sequence modeling | Dropout, L2 regularization, Early stopping |
| **XGBoost with Optuna** | Gradient Boosting | Automated hyperparameter tuning, SHAP values | Walk-forward validation, Class imbalance handling |
| **Improved Statistical** | Rule-Based | Multi-factor risk scoring, Dynamic thresholds | Threshold calibration on historical data |

### How Models Work

**LSTM (Long Short-Term Memory)**:
- Analyzes 60-day sequences of market data
- Attention mechanism focuses on important time periods
- Best for: Detecting temporal patterns and trends

**XGBoost (Extreme Gradient Boosting)**:
- Automatically optimized with Optuna (100 trials)
- SHAP values explain each prediction
- Best for: Feature-driven predictions with interpretability

**Statistical Model**:
- Weighted risk factors (yield curve 25%, volatility 20%, etc.)
- Dynamic threshold calibration
- Best for: Quick analysis and baseline predictions

### Model Selection

All models are tracked in MLflow. You can:
1. Compare models in MLflow UI (http://localhost:5000)
2. View SHAP values for interpretability
3. Promote best models to Production stage
4. Track all experiments and hyperparameters

---

## 📈 Understanding the 20 Indicators

### Yield Curve Indicators (3)
- **10Y-3M Spread**: Long-term vs. short-term rates (negative = recession signal)
- **10Y-2Y Spread**: Treasury yield curve (inverted = warning)
- **10-Year Yield**: Benchmark interest rate

### Credit Indicators (1)
- **BBB Credit Spread**: Corporate bond risk premium (rising = credit stress)

### Economic Indicators (5)
- **Unemployment Rate**: Labor market health (rising = weakness)
- **Real GDP**: Economic growth (declining = recession)
- **CPI**: Inflation rate (high = Fed tightening risk)
- **Fed Funds Rate**: Central bank policy rate
- **Industrial Production**: Manufacturing activity

### Market Indicators (3)
- **S&P 500 Price**: Stock market level
- **S&P 500 Volume**: Trading activity
- **VIX Index**: Volatility/fear gauge (>30 = elevated fear)

### Sentiment Indicators (1)
- **Consumer Sentiment**: Economic optimism

### Housing Indicators (1)
- **Housing Starts**: New construction activity

### Monetary Indicators (1)
- **M2 Money Supply**: Money in circulation

### Debt Indicators (1)
- **Debt to GDP**: Government debt burden

### Savings Indicators (1)
- **Savings Rate**: Consumer savings behavior

### Composite Indicators (1)
- **Leading Economic Index**: Forward-looking composite

### Alternative Indicators (2)
- **Margin Debt**: Market leverage (real data from FINRA or synthetic proxy)
- **Put/Call Ratio**: Options sentiment (real data from CBOE or synthetic proxy)

---

## 🔧 Advanced Configuration

### PostgreSQL Migration (Production)

For production use, migrate to PostgreSQL:

```bash
# 1. Install PostgreSQL
brew install postgresql  # macOS
# or
sudo apt-get install postgresql  # Linux

# 2. Create database
createdb crash_predictor

# 3. Update .env
POSTGRESQL_URL=postgresql://user:password@localhost:5432/crash_predictor

# 4. Run migration
python scripts/database/migrate_to_postgresql.py

# 5. Update DATABASE_URL
DATABASE_URL=postgresql://user:password@localhost:5432/crash_predictor
```

### Model Training Parameters

Edit `.env` to customize model training:

```bash
# LSTM Parameters
LSTM_UNITS=128              # Number of LSTM units per layer
LSTM_LAYERS=3               # Number of LSTM layers
LSTM_DROPOUT=0.3            # Dropout rate (anti-overfitting)
LSTM_SEQUENCE_LENGTH=60     # Days of history to analyze
LSTM_BATCH_SIZE=32          # Training batch size
LSTM_EPOCHS=100             # Maximum training epochs
LSTM_LEARNING_RATE=0.001    # Learning rate

# XGBoost Optimization
OPTUNA_N_TRIALS=100         # Number of Optuna trials
OPTUNA_TIMEOUT=3600         # Timeout in seconds (1 hour)

# Walk-Forward Validation
WALK_FORWARD_WINDOW_SIZE=252      # Training window (1 year)
WALK_FORWARD_STEP_SIZE=21         # Step size (1 month)
WALK_FORWARD_MIN_TRAIN_SIZE=1260  # Minimum training size (5 years)
```

---

## 🐛 Troubleshooting

### "FRED_API_KEY not set" Error
```bash
# 1. Check if .env file exists
ls -la .env

# 2. If not, create it
cp .env.example .env

# 3. Edit and add your FRED API key
nano .env

# 4. Verify it's set correctly
python -c "from src.utils.config import validate_api_keys; validate_api_keys()"
```

### Synthetic Data Warnings
```
⚠️ WARNING: Using synthetic Put/Call Ratio (CBOE_API_KEY not set)
⚠️ WARNING: Using synthetic Margin Debt (FINRA_API_KEY not set)
```

**This is NOT an error!** The system will work fine with synthetic proxies for these 2 indicators.

To remove warnings, add CBOE and FINRA API keys to `.env` (see API Keys section above).

### Dashboard not loading?
```bash
# Kill existing process
pkill -f streamlit

# Restart
streamlit run src/dashboard/app.py
```

### API not starting?
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Restart API
uvicorn src.api.main:app --reload
```

### Models not training?
```bash
# Check data collection first
python scripts/collect_data.py

# Then train models
python scripts/training/train_advanced_models.py

# Check logs
tail -f data/logs/training.log
```

### Database connection errors?
```bash
# Verify database exists
ls -la data/market_crash.db

# Test connection
python -c "from src.utils.database import DatabaseManager; db = DatabaseManager(); print('OK')"
```

### MLflow UI not showing experiments?
```bash
# Check MLflow directory
ls -la data/mlflow

# Start MLflow UI with correct path
mlflow ui --backend-store-uri data/mlflow
```

---

## 📞 Support & Documentation

### Documentation Files
- **README.md** - Main documentation with setup and usage
- **CHANGELOG.md** - Version history and migration guide
- **IMPLEMENTATION_SUMMARY.md** - Detailed implementation summary
- **QUICK_REFERENCE.md** - Quick reference for common tasks
- **docs/ARCHITECTURE.md** - System architecture
- **docs/METHODOLOGY.md** - Technical methodology

### Logs
Check logs for detailed error messages:
```bash
# Application logs
tail -f data/logs/app.log

# Training logs
tail -f data/logs/training.log

# Data collection logs
tail -f data/logs/data_collection.log
```

### Health Checks
```bash
# API health check
curl http://localhost:8000/health

# Database health check
python -c "from src.utils.database import DatabaseManager; db = DatabaseManager(); print('OK')"

# Configuration validation
python -c "from src.utils.config import validate_config; print(validate_config())"
```

---

## 🎯 Next Steps

### After Setup
1. ✅ **Collect Data**: `python scripts/collect_data.py`
2. ✅ **Train Models**: `python scripts/training/train_advanced_models.py`
3. ✅ **View MLflow**: `mlflow ui --backend-store-uri data/mlflow`
4. ✅ **Start API**: `uvicorn src.api.main:app --reload`
5. ✅ **Start Dashboard**: `streamlit run src/dashboard/app.py`

### Recommended Workflow
1. **Daily**: Run data collection to get latest indicators
2. **Weekly**: Retrain models with new data
3. **Monthly**: Review model performance in MLflow
4. **As Needed**: Promote best models to Production stage

### Production Deployment
1. **Migrate to PostgreSQL** (see Advanced Configuration)
2. **Set up automated backups** (already configured in `.env`)
3. **Enable monitoring** (Prometheus metrics at `/metrics`)
4. **Configure alerts** (set `ALERT_THRESHOLD` in `.env`)

---

## 🚀 Quick Command Reference

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env

# Data & Training
python scripts/collect_data.py
python scripts/training/train_advanced_models.py

# Services
mlflow ui --backend-store-uri data/mlflow          # MLflow UI
uvicorn src.api.main:app --reload                  # API Server
streamlit run src/dashboard/app.py                 # Dashboard

# Database
python scripts/database/migrate_to_postgresql.py   # PostgreSQL migration
python -c "from src.utils.backup import DatabaseBackup; DatabaseBackup().create_backup()"  # Backup

# Testing
pytest                                              # Run all tests
pytest --cov=src                                   # With coverage

# Monitoring
curl http://localhost:8000/health                  # Health check
curl http://localhost:8000/metrics                 # Prometheus metrics
```

---

## ✅ Checklist

Before running the system, make sure:

- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created from `.env.example`
- [ ] **FRED_API_KEY added to `.env`** (REQUIRED)
- [ ] CBOE_API_KEY added to `.env` (optional, for real Put/Call Ratio)
- [ ] FINRA_API_KEY added to `.env` (optional, for real Margin Debt)
- [ ] Data collected (`python scripts/collect_data.py`)
- [ ] Models trained (`python scripts/training/train_advanced_models.py`)

---

**System Status**: ✅ Production Ready v2.0
**Last Updated**: November 7, 2025

