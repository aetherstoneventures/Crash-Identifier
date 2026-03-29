# Market Crash & Bottom Prediction System - Architecture

**Last Updated**: October 25, 2025
**Status**: Production Ready

## 1. System Overview

A production-grade machine learning system for predicting market crashes and recovery points. The system combines multiple ML models with statistical validation to provide robust, interpretable predictions.

### Core Components
- **5 ML Models** for crash prediction (SVM, Random Forest, Gradient Boosting, Neural Network, Ensemble)
- **2 ML Models** for bottom prediction (MLP, LSTM)
- **1 Statistical Model** for rule-based validation
- **28 Financial Indicators** from FRED, Yahoo Finance, and alternative sources
- **11,430+ Historical Records** (1982-2025)
- **Streamlit Dashboard** for real-time visualization
- **SQLite Database** for persistent storage

## 2. Data Pipeline Architecture

### 2.1 Data Collection (`src/data_collection/`)
- **FREDCollector**: Fetches 20+ economic indicators from Federal Reserve Economic Data
- **YahooCollector**: Fetches S&P 500 and VIX price data
- **AlternativeCollector**: Fetches alternative data sources (margin debt, put/call ratio, etc.)

### 2.2 Feature Engineering (`src/feature_engineering/`)
- **FeaturePipeline**: Calculates 28 financial indicators:
  - Market Data: S&P 500, VIX
  - Yield Spreads: 10Y-3M, 10Y-2Y, 10Y
  - Economic: Unemployment, GDP, Industrial Production, CPI
  - Sentiment: Consumer Sentiment, Credit Spreads, Margin Debt, Put/Call Ratio
  - And 15 more...

### 2.3 Data Storage (`src/utils/database.py`)
- **SQLite Database**: `data/market_crash.db`
- **Tables**:
  - `indicators`: 28 financial indicators per date
  - `predictions`: ML model predictions with confidence intervals
  - `crash_events`: Historical crash events (20% drawdown)
  - `alert_history`: Alert logs

## 3. Model Architecture

### 3.1 Crash Prediction Models (`src/models/crash_prediction/`)

**BaseCrashModel**: Abstract base class defining interface
- `train(X_train, y_train, X_val, y_val)`: Train model
- `predict(X)`: Binary crash prediction (0 or 1)
- `predict_proba(X)`: Crash probability (0-1)
- `get_metrics()`: Model performance metrics

**ML Models**:
1. **SVMCrashModel**: Support Vector Machine with RBF kernel
2. **RandomForestCrashModel**: 100 trees, max_depth=15
3. **GradientBoostingCrashModel**: 100 estimators, learning_rate=0.1
4. **NeuralNetworkCrashModel**: 3-layer MLP (128-64-32 neurons)
5. **EnsembleCrashModel**: Weighted voting of all 4 models

**Statistical Model**:
- **StatisticalCrashModel**: Rule-based predictions using indicator thresholds
  - Inverted yield curve (10Y-2Y < 0) → 30% risk
  - High VIX (> 30) → 15-25% risk
  - High Shiller PE (> 30) → 10-15% risk
  - Unemployment spike (> 5%) → 10-15% risk
  - High credit spreads (BBB > 3%) → 5-10% risk
  - Margin debt → 5% risk

### 3.2 Bottom Prediction Models (`src/models/bottom_prediction/`)

**BaseBottomModel**: Abstract base class
- `train(X_train, y_train, X_val, y_val)`: Train model
- `predict(X)`: Predict days to bottom (regression)
- `get_metrics()`: Model performance metrics

**ML Models**:
1. **MLPBottomModel**: 3-layer MLP for regression
2. **LSTMBottomModel**: LSTM for time-series regression

### 3.3 Crash Labeling (`src/models/crash_prediction/crash_labeler.py`)
- **CrashLabeler**: Generates binary labels for training
  - Threshold: 20% drawdown from peak
  - Lookforward window: 60 days
  - Identifies historical crash periods

### 3.4 Bottom Labeling (`src/models/bottom_prediction/bottom_labeler.py`)
- **BottomLabeler**: Generates regression labels
  - Calculates days from crash to market bottom
  - Used for training bottom prediction models

## 4. Dashboard Architecture (`src/dashboard/app.py`)

### 4.1 Tabs (Clean UI Design)
1. **📊 Overview**: Summary metrics, S&P 500 price history, latest market data
2. **🚨 Crash Predictions**: ML vs Statistical comparison, probability metrics, prediction history
3. **📈 Bottom Predictions**: Days to market bottom, recovery timing, prediction statistics
4. **📋 Indicators**: Economic indicators, sentiment indicators, individual charts with normalized scaling

### 4.2 Key Features
- **ML vs Statistical Comparison**: Validate ML predictions against rule-based statistical model
- **Normalized Indicator Scaling**: Multiple indicators displayed on 0-100 scale for clarity
- **Error Handling**: Graceful error messages for missing or invalid data
- **Confidence Intervals**: Calculated from ensemble model disagreement
- **Caching**: 5-minute TTL for optimal performance
- **Clean UI**: No unnecessary sidebar or menu elements

## 5. Alert System (`src/alerts/`)
- **AlertManager**: Sends alerts when crash probability exceeds threshold
- **Channels**: Email, SMS (optional)
- **AlertHistory**: Logs all alerts in database

## 6. Scheduler (`src/scheduler/daily_tasks.py`)
- **DailyTaskScheduler**: Runs daily at 6 AM (configurable)
- **Tasks**:
  1. Collect latest data from FRED, Yahoo, alternatives
  2. Calculate features
  3. Generate predictions
  4. Store in database
  5. Send alerts if needed

## 7. Testing (`tests/`)
- **154 Tests**: 100% pass rate
- **Coverage**:
  - Model training and prediction
  - Feature engineering
  - Data collection and validation
  - Database operations
  - Alert system
  - Full pipeline integration

## 8. Quick Start

### 8.1 Run Full Pipeline
```bash
bash scripts/run_pipeline.sh
```
This:
1. Creates virtual environment
2. Installs dependencies
3. Runs data backfill (1982-2025)
4. Trains all 7 models
5. Runs 154 tests
6. Starts dashboard

### 8.2 Run Dashboard Only
```bash
bash scripts/run_dashboard.sh
```

### 8.3 Access Dashboard
```
http://localhost:8501
```

## 9. System Capabilities

### 9.1 Prediction Models
- **Crash Prediction**: Ensemble of 5 models with 85-93% AUC
- **Bottom Prediction**: LSTM and MLP models for recovery timing
- **Statistical Validation**: Rule-based model for interpretability

### 9.2 Data & Indicators
- **Historical Data**: 11,430 records spanning 1982-2025
- **28 Financial Indicators**: Market, economic, sentiment, and valuation metrics
- **Real-time Updates**: Daily data collection from FRED, Yahoo Finance, alternative sources
- **Data Validation**: Automatic range checking and anomaly detection

## 10. Performance Metrics

- **Data**: 11,430 records (1982-2025)
- **Models**: 7 trained (5 crash + 2 bottom)
- **Tests**: 154 passing (100% pass rate)
- **Dashboard Load**: < 100ms
- **Cache Refresh**: 5 minutes
- **Predictions**: Vary from 0.000005 to 0.9998

## 11. File Structure

```
market-crash-predictor/
├── docs/
│   ├── ARCHITECTURE.md (this file)
│   └── initiation_docs/
├── scripts/
│   ├── run_pipeline.sh
│   ├── run_dashboard.sh
│   ├── train_models.py
│   └── backfill_data.py
├── src/
│   ├── models/
│   │   ├── crash_prediction/
│   │   │   ├── statistical_model.py
│   │   │   ├── ensemble_model.py
│   │   │   └── (4 other ML models)
│   │   └── bottom_prediction/
│   ├── dashboard/
│   │   └── app.py
│   ├── data_collection/
│   ├── feature_engineering/
│   ├── alerts/
│   ├── scheduler/
│   └── utils/
├── tests/
├── data/
│   ├── market_crash.db
│   ├── models/
│   ├── processed/
│   ├── raw/
│   └── logs/
├── README.md
└── requirements.txt
```

## 12. Future Enhancements

1. **Model Retraining**: Implement automatic retraining pipeline with new data
2. **Advanced Analytics**: Add correlation analysis and feature importance visualization
3. **Backtesting Framework**: Comprehensive historical performance analysis
4. **API Layer**: REST API for programmatic access to predictions
5. **Alert Notifications**: Email and SMS alerts for high-probability events

---

**For setup instructions and quick start guide, see README.md**

