# Documentation Index

> Production reference for the current system:
> - **[V5_HONEST_SCORECARD.md](V5_HONEST_SCORECARD.md)** — what v5 actually does on BLIND (near-coincident regime detector, median lead −9d)
> - **[FORWARD_RISK.md](FORWARD_RISK.md)** — 1-month probabilistic forecast (only h=21 ships; h=63/126/252 SHELVED)
> - **[FUTURE_WORK_RESULTS.md](FUTURE_WORK_RESULTS.md)** — v5.1 / v6 / v5_multi all FAILED BLIND kill criteria
> - **[V6_NEGATIVE_RESULT.md](V6_NEGATIVE_RESULT.md)** — predictive-label experiment, killed
>
> The files below are kept for historical context but are **stale (pre-v5 era)**.

## 🚀 Quick Start
- **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** - Get started in 5 minutes

## 🏗️ Architecture & Design (historical)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Legacy 5-ML system architecture
- **[METHODOLOGY.md](METHODOLOGY.md)** - Legacy statistical / GB+RF methodology

## 📚 Reference Documentation
- **[HISTORICAL_CRASHES_REFERENCE.md](HISTORICAL_CRASHES_REFERENCE.md)** - 11 documented market crashes (1980-2022)
- **[REPRODUCIBILITY_GUIDE.md](REPRODUCIBILITY_GUIDE.md)** - (historical) v2 pipeline reproduction

---

## 📊 System Overview

### Data Coverage
- **20 High-Quality Indicators** with 100% data coverage (1982-2025)
- **11 Historical Crashes** documented and validated
- **11,434 Daily Records** with perfect continuity

### Model Performance
- **ML Model V5**: 81.8% recall (9/11 crashes detected), no overfitting
- **Statistical Model V2**: 81.8% recall (9/11 crashes detected)
- **Bottom Predictor**: ML-based optimal re-entry timing

### Key Features
- ✅ Real-time crash probability predictions
- ✅ Optimal market re-entry timing (bottom predictions)
- ✅ Interactive Streamlit dashboard
- ✅ Fully reproducible pipeline

---

## 📁 File Organization

```
docs/
├── README.md (this file)
├── QUICK_START_GUIDE.md - Get started in 5 minutes
├── ARCHITECTURE.md - System architecture
├── METHODOLOGY.md - Prediction methodology
├── HISTORICAL_CRASHES_REFERENCE.md - 11 documented crashes
├── MODEL_SELECTION_FAQ.md - FAQ about models
└── REPRODUCIBILITY_GUIDE.md - Reproduce all results
```

---

## 📖 Recommended Reading Order

1. **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** - Start here!
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design
3. **[METHODOLOGY.md](METHODOLOGY.md)** - How predictions work
4. **[HISTORICAL_CRASHES_REFERENCE.md](HISTORICAL_CRASHES_REFERENCE.md)** - Crash data
5. **[REPRODUCIBILITY_GUIDE.md](REPRODUCIBILITY_GUIDE.md)** - Reproduce results
6. **[MODEL_SELECTION_FAQ.md](MODEL_SELECTION_FAQ.md)** - Common questions

---

## 🎯 Production Recommendations

1. **Use both models** (ML + Statistical) for redundancy
2. **Monitor rate-of-change** indicators for early warning
3. **Review predictions daily** during high-risk periods
4. **Validate against market conditions** and news
5. **Use bottom predictions** for optimal re-entry timing

---

## ❓ Questions?

Refer to [MODEL_SELECTION_FAQ.md](MODEL_SELECTION_FAQ.md) for common questions about model selection and usage.
