# Market Crash Predictor - Documentation Index

## 📚 Quick Navigation

### Getting Started
- **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** - How to use the dashboard and run the pipeline
- **[README.md](../README.md)** - Project overview and installation

### System Architecture & Design
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design, components, and data flow
- **[METHODOLOGY.md](METHODOLOGY.md)** - Mathematical models and algorithms

### Model Documentation
- **[MODEL_IMPROVEMENTS_SUMMARY.md](MODEL_IMPROVEMENTS_SUMMARY.md)** - Advanced models and performance metrics
- **[MODEL_SELECTION_FAQ.md](MODEL_SELECTION_FAQ.md)** - How models are selected and weighted
- **[ACCURACY_IMPROVEMENTS.md](ACCURACY_IMPROVEMENTS.md)** - Performance optimization details

### Data & Validation
- **[VALIDATION_RESULTS.md](VALIDATION_RESULTS.md)** - Data quality and validation metrics
- **[DATA_QUALITY_FIX.md](DATA_QUALITY_FIX.md)** - Data range corrections and fixes

### Status & Reports
- **[FINAL_STATUS_REPORT.md](FINAL_STATUS_REPORT.md)** - Complete system status and verification
- **[PIPELINE_EXECUTION_SUMMARY.md](PIPELINE_EXECUTION_SUMMARY.md)** - Pipeline execution details
- **[TEST_FIX_SUMMARY.md](TEST_FIX_SUMMARY.md)** - Test suite fixes and results

### Legacy Documentation
- **[initiation_docs/](initiation_docs/)** - Original project initiation documents

---

## 🎯 By Use Case

### I want to...

**...use the dashboard**
→ Read [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)

**...understand the models**
→ Read [METHODOLOGY.md](METHODOLOGY.md) and [MODEL_IMPROVEMENTS_SUMMARY.md](MODEL_IMPROVEMENTS_SUMMARY.md)

**...check data quality**
→ Read [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md) and [DATA_QUALITY_FIX.md](DATA_QUALITY_FIX.md)

**...understand system architecture**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md)

**...verify system status**
→ Read [FINAL_STATUS_REPORT.md](FINAL_STATUS_REPORT.md)

**...understand model selection logic**
→ Read [MODEL_SELECTION_FAQ.md](MODEL_SELECTION_FAQ.md)

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| Database Records | 11,430 (1982-2025) |
| Indicators | 28 calculated + 18 raw |
| ML Model AUC | 0.9999-1.0000 |
| Statistical Model | Dynamic thresholds |
| Test Coverage | 154 passing tests |
| Data Quality | 100% valid |

---

## 🔧 System Components

### Data Collection (`src/data_collection/`)
- FRED API integration for economic indicators
- Yahoo Finance integration for market data
- Data validation and cleaning

### Feature Engineering (`src/feature_engineering/`)
- 28 crash prediction indicators
- Regime detection (normal, stress, crisis)
- Feature normalization and redundancy removal

### Models (`src/models/`)
- **Crash Prediction**: ML ensemble + statistical models
- **Bottom Prediction**: MLP and LSTM neural networks
- **Advanced Models**: Ensemble with SMOTE and stacking

### Dashboard (`src/dashboard/`)
- Streamlit-based interactive interface
- Real-time predictions and alerts
- Data validation and quality metrics

### Alerts (`src/alerts/`)
- Alert generation and history tracking
- Notification system

---

## 📈 Performance Summary

### Crash Prediction Models
| Model | Type | AUC | Status |
|-------|------|-----|--------|
| Advanced Ensemble | ML | 0.9999 | ✅ Active |
| Advanced Statistical | Rule-based | Dynamic | ✅ Active |
| Random Forest | ML | 0.9652 | ✅ Trained |
| Gradient Boosting | ML | 0.9719 | ✅ Trained |

### Bottom Prediction Models
| Model | Type | Status |
|-------|------|--------|
| MLP | Neural Network | ✅ Trained |
| LSTM | Recurrent | ✅ Trained |

---

## 🚀 Quick Commands

```bash
# Run the pipeline
bash scripts/run_pipeline.sh

# Start the dashboard
bash scripts/run_dashboard.sh

# Run tests
python3 -m pytest tests/ -v

# Train models
python3 scripts/train_models.py

# Validate data
python3 scripts/validate_predictions.py
```

---

## 📞 Support

For issues or questions:
1. Check the relevant documentation file above
2. Review [FINAL_STATUS_REPORT.md](FINAL_STATUS_REPORT.md) for system status
3. Check test results in [TEST_FIX_SUMMARY.md](TEST_FIX_SUMMARY.md)

---

**Last Updated**: October 26, 2025  
**Status**: ✅ Production Ready

