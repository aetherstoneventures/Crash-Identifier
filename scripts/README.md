# Scripts Directory

This directory contains all the scripts for the Market Crash & Bottom Prediction System.

## 📁 Directory Structure

```
scripts/
├── data/                    # Data collection and setup
│   ├── collect_data.py
│   └── populate_crash_events.py
├── training/                # Model training
│   ├── train_crash_detector_v5.py
│   ├── train_statistical_model_v2.py
│   └── train_bottom_predictor.py
├── utils/                   # Prediction generation
│   ├── generate_predictions_v5.py
│   └── generate_bottom_predictions.py
├── evaluation/              # Model evaluation
│   ├── evaluate_crash_detection.py
│   └── evaluate_bottom_predictions.py
├── run_pipeline.sh          # Full pipeline runner
└── run_dashboard.sh         # Dashboard-only runner
```

---

## 📊 Script Descriptions

### 🔵 Data Collection (`data/`)

**`collect_data.py`**
- Collects 20 indicators from FRED API and Yahoo Finance
- Date range: 1982-01-01 to present
- Stores in SQLite database (`data/market_crash.db`)
- Calculates synthetic indicators (margin_debt, put_call_ratio)

**`populate_crash_events.py`**
- Populates `crash_events` table with 11 historical crashes
- Includes: start date, trough date, recovery date, max drawdown
- Used for model training and evaluation

---

### 🎓 Model Training (`training/`)

**`train_crash_detector_v5.py`**
- Trains ML crash detector (Gradient Boosting + Random Forest ensemble)
- 39 engineered features from 20 base indicators
- 5-Fold stratified cross-validation (anti-overfitting)
- Saves model to `data/models/crash_detector_v5.pkl`
- Performance: 81.8% recall, AUC 0.7323

**`train_statistical_model_v2.py`**
- Trains rule-based statistical model
- 6 risk factors with weighted scoring
- No training data required (rule-based)
- Saves model to `data/models/statistical_model_v2.pkl`
- Performance: 81.8% recall

**`train_bottom_predictor.py`**
- Trains bottom prediction model (Gradient Boosting)
- Predicts days to bottom and recovery days
- Trained on 11 historical crashes
- Saves model to `data/models/bottom_predictor.pkl`

---

### 🔮 Prediction Generation (`utils/`)

**`generate_predictions_v5.py`**
- Generates crash probability predictions for all dates
- Uses both ML V5 and Statistical V2 models
- Stores predictions in `predictions` table
- Includes rate-of-change calculations (1d, 5d, 20d)

**`generate_bottom_predictions.py`**
- Generates bottom predictions for all dates
- Predicts optimal re-entry timing
- Uses trained bottom predictor model
- Stores in database for dashboard display

---

### 📈 Evaluation (`evaluation/`)

**`evaluate_crash_detection.py`**
- Evaluates crash detection performance on 11 historical crashes
- Calculates recall, precision, advance warning days
- Generates performance reports
- Used in pipeline to verify model quality

**`evaluate_bottom_predictions.py`**
- Evaluates bottom prediction accuracy
- Compares predicted vs actual days to bottom/recovery
- Calculates MAE, RMSE metrics
- Validates bottom predictor performance

---

## 🚀 Usage

### Full Pipeline (Fresh Start)
```bash
bash scripts/run_pipeline.sh
```

**This will:**
1. ✅ Create virtual environment
2. ✅ Install dependencies
3. ✅ Collect data (20 indicators, 1982-2025)
4. ✅ Populate crash events (11 crashes)
5. ✅ Train all models (ML V5, Statistical V2, Bottom Predictor)
6. ✅ Generate predictions
7. ✅ Evaluate performance
8. ✅ Start dashboard on http://localhost:8501

### Dashboard Only
```bash
bash scripts/run_dashboard.sh
```

**Use this when:**
- Data and models already exist
- You just want to view the dashboard
- No retraining needed

---

## 🔄 Pipeline Execution Order

1. **Data Collection** → `data/collect_data.py`
2. **Crash Events** → `data/populate_crash_events.py`
3. **Train ML Model** → `training/train_crash_detector_v5.py`
4. **Train Statistical Model** → `training/train_statistical_model_v2.py`
5. **Train Bottom Predictor** → `training/train_bottom_predictor.py`
6. **Generate Crash Predictions** → `utils/generate_predictions_v5.py`
7. **Generate Bottom Predictions** → `utils/generate_bottom_predictions.py`
8. **Evaluate Performance** → `evaluation/evaluate_crash_detection.py`
9. **Start Dashboard** → `streamlit run src/dashboard/app.py`

---

## ⚙️ Configuration

All scripts use:
- **Database**: `data/market_crash.db` (SQLite)
- **Models**: `data/models/` directory
- **Config**: `src/utils/config.py`
- **Python**: 3.8+ (tested on 3.13)

---

## 📝 Notes

- All old/deprecated scripts (v3, v4) have been removed
- Current system uses only V5 ML and V2 Statistical models
- Scripts are organized by function for better maintainability
- All paths in `run_pipeline.sh` have been updated to new structure
