# Configuration Summary: Fresh Inference Data for Nov 9-11, 2025

## Problem Statement
Training data only goes up to a certain date and does NOT include Nov 9-11, 2025. For accurate inference, we need:
1. **Fresh data** collected specifically for the test period
2. **Historical context** (168-hour lookback) before the test period
3. **Same preprocessing** as training data
4. **Training scalers** to properly scale inference data

## Solution Architecture

### Data Collection Strategy
- **Collection Period**: Nov 2-11, 2025 (10 days = 240 hours)
  - Nov 2-8: 168 hours for sequence lookback
  - Nov 9-11: 72 hours for actual predictions
- **API Source**: Open-Meteo (air quality + weather)
- **Same Parameters**: Exact same features as training data

### Sequence Creation Logic
```
For predicting hour H:
  Input: [H-168 to H-1] (168 hours of history)
  Output: H (1 hour ahead prediction)

Example for Nov 9, 2025 00:00:
  Input: Nov 2 00:00 to Nov 8 23:00
  Output: Nov 9 00:00
```

This is why we need data from Nov 2, not just Nov 9.

### Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│  TRAINING PHASE (Already Completed)                         │
├─────────────────────────────────────────────────────────────┤
│  1. Collect data: Jan 2022 - Nov 2025 (not including 9-11) │
│  2. Preprocess: 150+ features                                │
│  3. Create sequences: 168-hour lookback                      │
│  4. Train 8 models                                           │
│  5. Save: Models + Scalers                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  INFERENCE PHASE (New)                                       │
├─────────────────────────────────────────────────────────────┤
│  1. Collect fresh data: Nov 2-11, 2025 (240 hours)         │
│  2. Preprocess: SAME 150+ features                          │
│  3. Create sequences: SAME 168-hour lookback                │
│  4. Load trained models + scalers                            │
│  5. Scale with TRAINING scalers (not new scalers)           │
│  6. Predict on Nov 9-11 (72 predictions)                    │
│  7. Inverse transform with TRAINING scalers                  │
│  8. Calculate metrics, visualize                             │
└─────────────────────────────────────────────────────────────┘
```

## Files Created/Modified

### New Files for Inference

1. **collect_inference_data.py**
   - Fetches Nov 2-11, 2025 data from Open-Meteo API
   - Same parameters as training data collection
   - Output: `raw_pm25_firozabad_inference_nov9_11.csv` (240 hours)

2. **preprocess_inference_data.py**
   - Applies exact same preprocessing as training
   - All 150+ features: temporal, industrial, secondary pollutants, atmospheric stability
   - Output: `cleaned_pm25_firozabad_inference_nov9_11.csv`

3. **INFERENCE_WORKFLOW.md**
   - Complete step-by-step guide
   - Troubleshooting section
   - Expected metrics and validation checklist

### Modified Files

4. **modal_inferencing.py**
   - **Before**: Tried to filter Nov 9-11 from test set (doesn't exist in training)
   - **After**: Loads fresh inference CSV, creates sequences, uses training scalers
   - Key changes:
     - Load `cleaned_pm25_firozabad_inference_nov9_11.csv` from Modal volume
     - Load training scalers from `firozabad_sequences_hourly_complete.pkl`
     - Create sequences on-the-fly from inference data
     - Scale using training scalers (transform only, no fit)
     - Works for both ML and DL models

## Critical Technical Details

### Scaling Must Use Training Scalers

❌ **WRONG** (Inference-only scalers):
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_inference)  # WRONG! Creates new scaler
```

✅ **CORRECT** (Training scalers):
```python
# Load training scaler
scaler = train_data['scaler_features']  # Fitted on training data
X_scaled = scaler.transform(X_inference)  # CORRECT! Uses training distribution
```

**Why?** Models were trained on a specific distribution. Inference data must be scaled to that same distribution, not its own distribution.

### Feature Engineering Must Match Exactly

Both training and inference must have:
- Same feature names
- Same feature order
- Same feature engineering logic
- Same handling of edge cases (first lag values, etc.)

The `preprocess_inference_data.py` ensures this by using identical functions.

### Sequence Creation on Fresh Data

```python
# For each prediction hour (starting from hour 168):
for i in range(LOOKBACK, len(data)):
    sequence = data[i-LOOKBACK:i]  # 168 hours before
    target = data[i]                # Current hour to predict
```

From 240 hours of data:
- Hours 0-167: Used only for creating first sequence
- Hours 168-239: 72 predictions (Nov 9-11)

## Execution Steps

### Local Execution
```bash
# Step 1: Collect fresh data
python collect_inference_data.py
# Output: raw_pm25_firozabad_inference_nov9_11.csv

# Step 2: Preprocess
python preprocess_inference_data.py
# Output: cleaned_pm25_firozabad_inference_nov9_11.csv
```

### Modal Upload
```bash
# Step 3: Upload to Modal
modal volume put ai_ml_firozabad cleaned_pm25_firozabad_inference_nov9_11.csv /
```

### Modal Execution
```bash
# Step 4: Run inference (on Modal H100)
modal run modal_inferencing.py
# This uses fresh data + trained models + training scalers
```

### Local Analysis
```bash
# Step 5: Download results
modal volume get ai_ml_firozabad inference_results_firozabad/ .

# Step 6: Visualize
python visualization.py
# Output: results_firozabad/ folder with plots
```

## Expected Behavior

### Data Collection Output
```
INFERENCE DATA COLLECTION - NOV 2-11, 2025
=========================================
Total records: 240
  Lookback data: 168 hours (Nov 2-8)
  Test data: 72 hours (Nov 9-11)
PM2.5 data:
  Available: 235 / 240 hours
  Missing: 5 hours (2.1%)
```

### Preprocessing Output
```
INFERENCE DATA PREPROCESSING - NOV 2-11, 2025
============================================
Final shape: (240, 155)
Total features: 154
  Temporal: 10
  Lag: 6
  Rolling: 16
  Industrial: 4
  Secondary pollutants: 3
  Atmospheric stability: 4
  Raw features: 111
```

### Inference Output (Per Model)
```
Running inference for firozabad - RandomForest...
  Loaded 240 inference records
  Loaded scalers from training data
  Features shape: (240, 154)
  Need 168 hours lookback for sequences
  Created 72 sequences
  Sequence shape: (72, 168, 154)
  Loaded model from checkpoints_firozabad/firozabad_RandomForest_model.joblib
✓ Inference complete: RMSE=8.4523, MAE=6.1234, R²=0.8456
```

## Validation Checklist

Before claiming success:

- [ ] Collected 240 hours (not just 72)
- [ ] Preprocessed with same features (check feature count matches training)
- [ ] Uploaded to correct Modal volume (`ai_ml_firozabad`)
- [ ] All 8 models completed without errors
- [ ] Metrics are reasonable (R² > 0, RMSE > 0)
- [ ] Results downloaded successfully
- [ ] Visualizations show sensible patterns (not random noise)

## Key Files Location

| File | Location | Size |
|------|----------|------|
| Raw inference data | Local | ~100 KB |
| Cleaned inference data | Local → Modal | ~500 KB |
| Training scalers | Modal volume | In .pkl file |
| Trained models (8) | Modal volume | ~50-200 MB each |
| Inference results | Modal → Local | ~2 MB |
| Visualizations | Local | ~10 MB total |

## Common Errors and Solutions

### "Not enough data for sequences"
- **Cause**: Only collected 72 hours instead of 240
- **Fix**: Collect from Nov 2-11 (not just Nov 9-11)

### "Feature count mismatch"
- **Cause**: Preprocessing different from training
- **Fix**: Check preprocessing script matches training exactly

### "Model checkpoint not found"
- **Cause**: Training not completed or wrong volume name
- **Fix**: Verify training completed, check volume name

### "Scaler error / values out of range"
- **Cause**: Using wrong scalers or fitting new scalers on inference data
- **Fix**: Load training scalers, use transform only (not fit_transform)

### "Negative R² or very bad metrics"
- **Cause**: Wrong scaling, feature mismatch, or model loaded incorrectly
- **Fix**: Verify scalers, features, and model architecture match training

## Performance Expectations

### Good Results
- RMSE: 5-15 μg/m³
- MAE: 3-10 μg/m³
- R²: 0.7-0.95
- MAPE: 10-25%

### Concerning Results
- RMSE > 30: Something wrong with scaling or features
- R² < 0: Model completely failed, check everything
- MAPE > 50%: Likely wrong scalers being used

## Next Steps After Successful Inference

1. **Model Selection**: Choose best model based on R² and RMSE
2. **Error Analysis**: Look at hours with high errors
3. **Diurnal Validation**: Check if model captures morning/evening peaks
4. **Deployment**: Use best model for real-time predictions
5. **Monitoring**: Track performance over time, retrain if needed

## Summary

The key insight is: **Training data ≠ Test data for Nov 9-11, 2025**

We need:
1. Fresh API call for Nov 2-11, 2025 (240 hours)
2. Same preprocessing pipeline
3. Training scalers (not new scalers!)
4. Sequence creation from fresh data
5. Predictions on hours 168-239 (Nov 9-11)

This approach ensures:
- ✅ Real-world test scenario
- ✅ Proper temporal dependencies (168-hour lookback)
- ✅ Correct scaling (using training distribution)
- ✅ Feature alignment (same features as training)
- ✅ Valid metrics (testing on truly unseen data)
