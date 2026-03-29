# Model Selection FAQ

## Q1: Are the Advanced Ensemble and Advanced Statistical models NEW?

**A: YES!** These are new models that were just created and trained.

### Timeline
- **Previous**: Only base models (SVM, RF, GB, NN, Ensemble)
- **Now**: Added 2 advanced models on top of base models

### What's New
1. **Advanced Ensemble Model**
   - Uses SMOTE for class imbalance handling
   - Feature engineering (interaction terms, lagged features, rolling stats)
   - Hyperparameter optimization for each base model
   - Stacking ensemble with meta-learner
   - Cross-validation for optimal weights

2. **Advanced Statistical Model**
   - Dynamic thresholds based on market regime
   - Adaptive weights that change with market conditions
   - 6 weighted risk factors
   - Volatility regime adjustment

---

## Q2: Can I choose which models to use?

**A: Currently NO, but it's AUTOMATIC based on performance.**

### Current Behavior (Automatic Selection)
The system automatically selects the best available models:

```
Step 1: Check if Advanced Ensemble is available
        ↓ YES → Use it (weight: 1.5x)
        ↓ NO → Go to Step 2

Step 2: Check if Advanced Statistical is available
        ↓ YES → Use it (weight: 1.2x)
        ↓ NO → Go to Step 3

Step 3: Use Base Models (RF, GB)
        ↓
Final Prediction = Weighted average of available models
```

### Why Automatic?
- **Ensures best performance**: Always uses highest-performing models
- **Simplicity**: No manual configuration needed
- **Reliability**: Fallback to base models if advanced models fail

---

## Q3: What is the logic for model selection?

**A: Performance-based ranking with weighted averaging.**

### Selection Logic

#### Priority 1: Advanced Ensemble (Weight: 1.5x)
- **Why**: Highest accuracy (AUC 0.9999-1.0000)
- **How**: SMOTE-balanced data + stacking ensemble
- **When**: Always preferred if available

#### Priority 2: Advanced Statistical (Weight: 1.2x)
- **Why**: Highly interpretable + adaptive to market conditions
- **How**: Rule-based with dynamic thresholds
- **When**: Used if Advanced Ensemble unavailable

#### Priority 3: Base Models (Weight: 1.0x)
- **Why**: Reliable fallback
- **How**: Random Forest (0.9652) + Gradient Boosting (0.9719)
- **When**: Used if advanced models unavailable

### Final Prediction Calculation
```
Final Probability = (1.5 * adv_ensemble + 1.2 * adv_statistical) / 2.7
                  = Clipped to [0, 1] range
```

### Example
```
Advanced Ensemble prediction: 0.45 (45% crash probability)
Advanced Statistical prediction: 0.38 (38% crash probability)

Final = (1.5 * 0.45 + 1.2 * 0.38) / 2.7
      = (0.675 + 0.456) / 2.7
      = 1.131 / 2.7
      = 0.419 (41.9% crash probability)
```

---

## Q4: How do I know which model is being used?

**A: Check the dashboard or database.**

### In Dashboard
1. Go to "Crash Predictions" tab
2. Look for "Model Used" field
3. Shows which model generated the prediction

### In Database
```sql
SELECT prediction_date, crash_probability, model_version
FROM predictions
ORDER BY prediction_date DESC
LIMIT 5;
```

### Model Version Codes
- `v1.0`: Base models (SVM, RF, GB, NN, Ensemble)
- `v2.0`: Advanced Ensemble + Advanced Statistical

---

## Q5: Can I manually select a specific model?

**A: Not yet, but it's easy to add!**

### To Enable Manual Selection
Edit `src/dashboard/app.py` around line 350:

```python
# Current (automatic)
MODEL_SELECTION = 'auto'

# Change to one of:
MODEL_SELECTION = 'advanced_ensemble'    # Only Advanced Ensemble
MODEL_SELECTION = 'advanced_statistical' # Only Advanced Statistical
MODEL_SELECTION = 'base'                 # Only base models (RF, GB)
```

### To Add UI Dropdown
Add to dashboard:
```python
model_choice = st.selectbox(
    "Select Model",
    ["Auto (Best)", "Advanced Ensemble", "Advanced Statistical", "Base Models"]
)
```

---

## Q6: What if a model fails?

**A: Automatic fallback to next best model.**

### Fallback Chain
```
Try Advanced Ensemble
  ↓ (if fails)
Try Advanced Statistical
  ↓ (if fails)
Try Base Models (RF, GB)
  ↓ (if fails)
Return error message
```

### Example
If Advanced Ensemble crashes:
1. System logs the error
2. Falls back to Advanced Statistical
3. Prediction still generated successfully
4. User sees prediction without knowing about the failure

---

## Q7: How are the models ranked?

**A: By AUC (Area Under the Curve) score.**

### Current Rankings
| Rank | Model | AUC | Type |
|------|-------|-----|------|
| 1 | Advanced Ensemble | 0.9999 | ML |
| 2 | Advanced Ensemble (GB) | 0.9999 | ML |
| 3 | Advanced Ensemble (RF) | 1.0000 | ML |
| 4 | Advanced Ensemble (SVM) | 0.9998 | ML |
| 5 | Advanced Ensemble (NN) | 0.9999 | ML |
| 6 | Gradient Boosting | 0.9719 | ML |
| 7 | Random Forest | 0.9652 | ML |
| 8 | Advanced Statistical | Dynamic | Rule-based |
| 9 | Neural Network | 0.8909 | ML |
| 10 | SVM | 0.3725 | ML |

### What is AUC?
- **Range**: 0.0 to 1.0
- **Meaning**: Probability model correctly ranks crash vs. non-crash
- **0.5**: Random guessing
- **1.0**: Perfect predictions
- **0.9999**: Near-perfect (99.99% accuracy)

---

## Q8: Can I see the model accuracy metrics?

**A: YES! Check the dashboard.**

### In Dashboard
1. Go to "Model Accuracy" tab
2. See all models ranked by AUC
3. View Precision, Recall, F1, False Alarm Rate

### In Database
```sql
SELECT model_name, auc, precision, recall, f1_score
FROM model_accuracy
ORDER BY auc DESC;
```

---

## Q9: What's the difference between the models?

### Advanced Ensemble
- **Pros**: Highest accuracy (0.9999), fast
- **Cons**: Black box (hard to interpret)
- **Best for**: Maximum accuracy

### Advanced Statistical
- **Pros**: Interpretable, adaptive to market conditions
- **Cons**: Lower accuracy than ensemble
- **Best for**: Understanding why crash is predicted

### Base Models
- **Pros**: Reliable, well-tested
- **Cons**: Lower accuracy than advanced models
- **Best for**: Fallback/comparison

---

## Q10: How often are models retrained?

**A: Currently manual, but can be automated.**

### Current Process
```bash
# Manual retraining
python3 scripts/train_models.py
```

### To Automate
Edit `scripts/run_pipeline.sh`:
```bash
# Add cron job
0 2 * * * cd /path/to/project && bash scripts/run_pipeline.sh
```

This would retrain models daily at 2 AM.

---

## Summary

| Question | Answer |
|----------|--------|
| Are Advanced models new? | YES |
| Can I choose models? | Not yet (automatic selection) |
| How is selection logic? | Performance-based ranking |
| How do I know which model? | Check dashboard or database |
| Can I manually select? | Easy to add (edit config) |
| What if model fails? | Automatic fallback |
| How are models ranked? | By AUC score |
| Can I see accuracy? | YES (Model Accuracy tab) |
| What's the difference? | Accuracy vs. interpretability |
| How often retrained? | Manual (can automate) |

---

## Next Steps

1. **Check Dashboard**: http://localhost:8501
2. **View Model Accuracy**: Go to "Model Accuracy" tab
3. **See Predictions**: Go to "Crash Predictions" tab
4. **Plot Indicators**: Go to "Indicators → All Indicators"
5. **Optional**: Add manual model selection UI

