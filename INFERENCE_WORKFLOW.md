# Firozabad PM2.5 Inference Workflow - Nov 9-11, 2025

## Overview
This document explains the complete workflow for running inference on fresh Nov 9-11, 2025 data using trained models.

**IMPORTANT**: The training data only goes up to a certain date (not including Nov 9-11, 2025). For inference, we need to:
1. Collect FRESH data for Nov 9-11, 2025 
2. Preprocess it with the SAME features as training
3. Use trained models and scalers from training
4. Generate predictions and metrics

---

## Complete Workflow

### Step 1: Collect Fresh Inference Data

```bash
python collect_inference_data.py
```

**What it does:**
- Fetches air quality and weather data for Nov 9-11, 2025 from Open-Meteo API
- Uses the same parameters as training data collection
- Saves to: `raw_pm25_firozabad_inference_nov9_11.csv`

**Expected output:**
- 72 hours of data (Nov 9 00:00 to Nov 11 23:59)
- Same feature set as training data
- May have some missing values (handled in preprocessing)

**NOTE:** Since we need 168-hour (7-day) lookback for sequences, you should ideally collect data from **Nov 2-11, 2025** (10 days total) to ensure you have enough historical context for all 72 prediction hours.

To do this, modify `collect_inference_data.py`:
```python
START_DATE = '2025-11-02'  # Changed from '2025-11-09'
END_DATE = '2025-11-11'
```

This gives you:
- Nov 2-8: 168 hours (7 days) for initial lookback
- Nov 9-11: 72 hours for actual predictions

---

### Step 2: Preprocess Inference Data

```bash
python preprocess_inference_data.py
```

**What it does:**
- Loads raw inference data
- Applies EXACT SAME preprocessing as training:
  - Handles missing values (interpolation, forward/backward fill)
  - Adds temporal features (hour, day, cyclical encoding)
  - Adds interaction features (temp×humidity, wind×pm10, etc.)
  - Adds industrial features (furnace hours, stagnation index, ventilation coefficient)
  - Adds secondary pollutant features (ozone formation, aerosol potential)
  - Adds atmospheric stability features (Richardson proxy, mixing depth)
  - Adds lag features (1, 2, 3, 6, 12, 24 hours)
  - Adds rolling features (mean, std, min, max for 3, 6, 12, 24 hour windows)
- Saves to: `cleaned_pm25_firozabad_inference_nov9_11.csv`

**Expected output:**
- Same number of rows as input (240 hours if you collected Nov 2-11)
- 150+ features (matching training data exactly)
- No missing values

---

### Step 3: Upload to Modal Volume

```bash
modal volume put ai_ml_firozabad cleaned_pm25_firozabad_inference_nov9_11.csv /
```

**What it does:**
- Uploads preprocessed inference data to Modal cloud storage
- Makes it accessible to inference functions running on Modal

**Verify upload:**
```bash
modal volume ls ai_ml_firozabad
```

You should see:
- `cleaned_pm25_firozabad_inference_nov9_11.csv` ✓
- `firozabad_sequences_hourly_complete.pkl` (training scalers) ✓
- `model_arch.py` (model definitions) ✓
- `checkpoints_firozabad/` (trained models) ✓

---

### Step 4: Run Inference on Modal

```bash
modal run modal_inferencing.py
```

**What it does:**
- Loads fresh inference data from Modal volume
- Loads trained scalers from training data (for proper scaling)
- Creates 168-hour sequences from inference data
- For each of 8 models:
  - Loads trained model checkpoint
  - Scales inference sequences using training scalers
  - Runs predictions
  - Inverse transforms to original scale
  - Calculates metrics (RMSE, MAE, R², MAPE, Variance)
- Saves results to Modal volume

**Models tested:**
1. RandomForest
2. XGBoost
3. LSTM (Standard)
4. GRU (Standard)
5. BiLSTM (Standard)
6. BiGRU (Standard)
7. Enhanced_BiLSTM (with Attention + Physics)
8. Enhanced_BiGRU (with Attention + Physics)

**Expected runtime:**
- ML models: ~30 seconds each
- Deep learning models: ~2-3 minutes each
- Total: ~20-25 minutes for all 8 models

**Output files (on Modal volume):**
- `inference_results_firozabad/firozabad_inference_nov9_11_2025.json` - Full predictions and targets
- `inference_results_firozabad/metrics_summary_nov9_11_2025.json` - Just metrics

---

### Step 5: Download Results

```bash
# Download everything
modal volume get ai_ml_firozabad inference_results_firozabad/ .

# Or download just the results file
modal volume get ai_ml_firozabad inference_results_firozabad/firozabad_inference_nov9_11_2025.json .
```

**What you get:**
- JSON file with predictions, actual values, timestamps, and metrics for all 8 models
- Metrics summary JSON with just the performance numbers

---

### Step 6: Visualize Results

```bash
python visualization.py
```

**What it does:**
- Loads inference results JSON
- Creates comprehensive visualizations:
  - Individual model analysis (8 files): Time series, scatter plots, diurnal patterns, metrics
  - Model comparison: Bar charts comparing RMSE, MAE, R², MAPE
  - Variance analysis: Actual vs predicted variance comparison
  - Performance summary: CSV table with all metrics
- Identifies best-performing models

**Output folder:** `results_firozabad/`

**Generated files:**
- `firozabad_{model}_nov9_11_analysis.png` (8 files)
- `model_comparison_nov9_11.png`
- `variance_comparison_nov9_11.png`
- `performance_summary_nov9_11.csv`

---

## Key Technical Details

### Sequence Creation Logic

For time-series models, we need a lookback window:
- **Lookback**: 168 hours (7 days)
- **Prediction horizon**: 1 hour ahead

Example for predicting Nov 9, 2025 at 00:00:
- Input sequence: Nov 2 00:00 to Nov 8 23:00 (168 hours)
- Target: Nov 9 00:00 (the hour we want to predict)

This is why we need data from Nov 2 onwards, not just Nov 9-11.

### Scaling Strategy

**Critical**: Inference data must be scaled using training scalers, NOT new scalers fit on inference data.

```python
# Load training scalers
scaler_features = train_data['scaler_features']  # Fitted on training data
scaler_target = train_data['scaler_target']      # Fitted on training targets

# Apply to inference data (transform only, no fit)
X_scaled = scaler_features.transform(X_inference)
y_scaled = scaler_target.transform(y_inference)

# After prediction, inverse transform
predictions_orig = scaler_target.inverse_transform(predictions)
```

### Feature Alignment

**Critical**: Inference data must have EXACT SAME features as training in SAME ORDER.

The preprocessing script ensures this by:
1. Using same feature engineering functions
2. Extracting features in same order
3. Handling missing features gracefully

### Model Loading

Each model type loads differently:

**ML Models** (RandomForest, XGBoost):
```python
model = joblib.load('firozabad_RandomForest_model.joblib')
predictions = model.predict(X_flattened)  # Sequences flattened to 2D
```

**Standard DL Models** (LSTM, GRU, BiLSTM, BiGRU):
```python
model = StandardLSTM(input_size, 256, 3, 0.3)
model.load_state_dict(torch.load('firozabad_LSTM_best.pth'))
predictions = model(X_sequences)  # 3D sequences [batch, time, features]
```

**Enhanced DL Models** (Enhanced_BiLSTM, Enhanced_BiGRU):
```python
model = EnhancedBiLSTMAttentionComplete(input_size, 256, 3, 0.3, 8)
model.load_state_dict(torch.load('firozabad_Enhanced_BiLSTM_best.pth'))
predictions, attention = model(X_sequences, physics_features)  # Needs physics dict
```

---

## Troubleshooting

### Problem: "Not enough data for sequences"
**Solution**: Collect data from Nov 2-11 (not just Nov 9-11) to have 168-hour lookback.

### Problem: "Model checkpoint not found"
**Solution**: Make sure training was completed and models saved to `checkpoints_firozabad/` in Modal volume.

### Problem: "Inference data not found"
**Solution**: Upload `cleaned_pm25_firozabad_inference_nov9_11.csv` to Modal volume.

### Problem: "Feature mismatch"
**Solution**: Ensure preprocessing script matches training exactly. Check feature count in both datasets.

### Problem: "Metrics seem wrong (negative R²)"
**Solution**: This happens if model completely fails. Check:
- Correct scalers being used (from training)
- Features in correct order
- Model loaded properly
- Enough inference data for sequences

### Problem: "Missing values in predictions"
**Solution**: Check if inference data had too many missing values. Improve handling in `preprocess_inference_data.py`.

---

## Validation Checklist

Before running inference, verify:

- [ ] Fresh data collected for Nov 2-11, 2025 (240 hours)
- [ ] Preprocessing completed with same features as training
- [ ] Preprocessed file has no missing values
- [ ] File uploaded to Modal volume `ai_ml_firozabad`
- [ ] Training was completed (8 model checkpoints exist)
- [ ] Training scalers exist in `firozabad_sequences_hourly_complete.pkl`
- [ ] `model_arch.py` uploaded to Modal volume

After inference:
- [ ] All 8 models completed successfully
- [ ] Metrics seem reasonable (R² > 0, RMSE > 0)
- [ ] Results downloaded locally
- [ ] Visualizations generated
- [ ] Best model identified

---

## Expected Results

### Reasonable Metrics for PM2.5 Prediction

- **RMSE**: 5-20 μg/m³ (lower is better)
- **MAE**: 3-15 μg/m³ (lower is better)
- **R²**: 0.5-0.95 (higher is better, >0.7 is good)
- **MAPE**: 10-30% (lower is better)

### Model Comparison Expectations

Typically:
- **XGBoost** > **RandomForest** (better at capturing non-linear patterns)
- **Enhanced models** > **Standard DL models** (attention + physics helps)
- **BiLSTM/BiGRU** > **LSTM/GRU** (bidirectional captures more context)
- **Deep learning** > **ML models** (for time-series with long dependencies)

But actual results depend on:
- Data quality for Nov 9-11
- How well training generalized
- Whether Nov 9-11 had unusual weather/pollution events

---

## Next Steps After Inference

1. **Analyze Results**: Look at visualizations, identify best model
2. **Error Analysis**: Check where predictions fail (morning vs evening, high vs low PM2.5)
3. **Feature Importance**: For ML models, check which features matter most
4. **Diurnal Patterns**: Verify models capture daily cycles correctly
5. **Extreme Events**: Check performance on high pollution hours
6. **Model Selection**: Choose best model for deployment based on metrics + interpretability

---

## Files Summary

| File | Purpose | When to Run |
|------|---------|-------------|
| `collect_inference_data.py` | Fetch fresh Nov 2-11, 2025 data | Step 1 (locally) |
| `preprocess_inference_data.py` | Feature engineering on fresh data | Step 2 (locally) |
| `modal_inferencing.py` | Run all 8 models on Modal | Step 4 (on Modal) |
| `visualization.py` | Create plots and analysis | Step 6 (locally) |

| Generated File | Contains | Location |
|----------------|----------|----------|
| `raw_pm25_firozabad_inference_nov9_11.csv` | Raw API data | Local |
| `cleaned_pm25_firozabad_inference_nov9_11.csv` | Preprocessed features | Local → Modal |
| `firozabad_inference_nov9_11_2025.json` | Predictions + metrics | Modal → Local |
| `metrics_summary_nov9_11_2025.json` | Just metrics | Modal → Local |
| `results_firozabad/*.png` | Visualizations | Local |
| `performance_summary_nov9_11.csv` | Metrics table | Local |

---

## Questions?

If you encounter issues not covered here:
1. Check Modal logs: `modal logs ai_ml_firozabad`
2. Verify file contents: `modal volume get ai_ml_firozabad <file> .`
3. Test locally first with a small subset
4. Check feature alignment between training and inference data

Good luck with your PM2.5 predictions! 🌫️📊
