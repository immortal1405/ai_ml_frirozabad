# Firozabad PM2.5 Inference Configuration

## Test Period: November 9-11, 2025

All inference is configured to specifically test on **November 9-11, 2025** data (72 hours) to provide accurate real-world test results.

## Configuration Changes

### 1. **Modal Volume Name Updated**
All files now use: `ai_ml_firozabad`

**Files updated:**
- ✓ `create_sequences.py`
- ✓ `modal_training.py`
- ✓ `modal_inferencing.py`

### 2. **Inference for All 8 Models**
The inference script will test all models:

#### ML Models (2)
1. **RandomForest**
2. **XGBoost**

#### Deep Learning Models (6)
3. **LSTM** (Standard)
4. **GRU** (Standard)
5. **BiLSTM** (Standard)
6. **BiGRU** (Standard)
7. **Enhanced_BiLSTM** (Your custom with attention + physics)
8. **Enhanced_BiGRU** (Your custom with attention + physics)

### 3. **Test Data Filtering**
The inference script automatically:
- Filters test data to **Nov 9-11, 2025** (00:00 to 23:59)
- Validates date range availability
- Falls back to full test set if Nov 9-11 data not available
- Reports number of test samples used

## Inference Process

### Phase 1: ML Models
```
Running RandomForest...
  Filtering to Nov 9-11, 2025
  Test samples: ~72 (1 per hour for 3 days)
  ✓ RMSE: X.XXXX, MAE: X.XXXX, R²: X.XXXX

Running XGBoost...
  [similar output]
```

### Phase 2: Deep Learning Models
```
Running LSTM...
  Filtering to Nov 9-11, 2025
  Test samples: ~72
  ✓ RMSE: X.XXXX, MAE: X.XXXX, R²: X.XXXX

[Continues for all 6 DL models]
```

## Output Files

Results are saved to Modal volume: `ai_ml_firozabad`

### Location
```
/data/inference_results_firozabad/
```

### Files Generated
1. **firozabad_inference_nov9_11_2025.json**
   - Complete predictions and targets for each model
   - Time series data with timestamps
   - Full metrics for each model

2. **metrics_summary_nov9_11_2025.json**
   - Quick comparison of all models
   - Only metrics (RMSE, MAE, R², MAPE, etc.)
   - Easy to parse for visualization

## Metrics Calculated

For each model on Nov 9-11, 2025 data:

1. **MSE** - Mean Squared Error
2. **RMSE** - Root Mean Squared Error (main metric)
3. **MAE** - Mean Absolute Error
4. **R²** - R-squared Score (goodness of fit)
5. **MAPE** - Mean Absolute Percentage Error
6. **Variance_Actual** - Variance of actual PM2.5 values
7. **Variance_Pred** - Variance of predicted values
8. **Variance_Diff** - Difference between actual and predicted variance

## Running Inference

### Step 1: Ensure Training is Complete
```bash
# Check if all 8 models are trained
modal volume ls ai_ml_firozabad checkpoints_firozabad/
```

You should see:
- `firozabad_RandomForest_model.joblib`
- `firozabad_XGBoost_model.joblib`
- `firozabad_LSTM_best.pth`
- `firozabad_GRU_best.pth`
- `firozabad_BiLSTM_best.pth`
- `firozabad_BiGRU_best.pth`
- `firozabad_Enhanced_BiLSTM_best.pth`
- `firozabad_Enhanced_BiGRU_best.pth`

### Step 2: Run Inference
```bash
modal run modal_inferencing.py
```

### Step 3: Download Results
```bash
# Download all results
modal volume get ai_ml_firozabad inference_results_firozabad/ .

# Or just the main JSON file
modal volume get ai_ml_firozabad inference_results_firozabad/firozabad_inference_nov9_11_2025.json .

# Or just the metrics summary
modal volume get ai_ml_firozabad inference_results_firozabad/metrics_summary_nov9_11_2025.json .
```

## Expected Output Format

### firozabad_inference_nov9_11_2025.json
```json
{
  "RandomForest": {
    "city": "firozabad",
    "model_type": "RandomForest",
    "test_period": "Nov 9-11, 2025",
    "num_samples": 72,
    "predictions": [23.4, 25.1, ...],
    "targets": [24.2, 26.3, ...],
    "time_index": ["2025-11-09 00:00:00", "2025-11-09 01:00:00", ...],
    "metrics": {
      "MSE": 15.234,
      "RMSE": 3.903,
      "MAE": 2.845,
      "R2": 0.876,
      "MAPE": 12.34,
      "Variance_Actual": 145.23,
      "Variance_Pred": 142.67,
      "Variance_Diff": 2.56
    }
  },
  "XGBoost": { ... },
  "LSTM": { ... },
  ...
}
```

### metrics_summary_nov9_11_2025.json
```json
{
  "RandomForest": {
    "MSE": 15.234,
    "RMSE": 3.903,
    "MAE": 2.845,
    "R2": 0.876,
    "MAPE": 12.34
  },
  "XGBoost": { ... },
  ...
}
```

## Why November 9-11, 2025?

This 3-day test period provides:
- **Recent data** - Most relevant for current conditions
- **72 hours** - Full diurnal cycles (3 complete days)
- **Variety** - Captures different times of day and pollution patterns
- **Real-world test** - True out-of-sample validation
- **Firozabad-specific** - Tests on actual Firozabad glass manufacturing pollution patterns

## Hardware Configuration

### Inference on Modal H100
- **GPU**: NVIDIA H100 (80GB)
- **CPU**: 8 cores
- **RAM**: 32GB
- **Batch Size**: 128
- **Platform**: Modal.com

## Troubleshooting

### Issue: No data for Nov 9-11, 2025
**Solution**: The script will automatically use the full test set and print a warning

### Issue: Model checkpoint not found
**Solution**: Ensure training completed successfully for all models

### Issue: Date range error
**Solution**: Check that your collected data includes dates up to Nov 11, 2025

## Next Steps After Inference

1. Download results JSON files
2. Analyze metrics to compare all 8 models
3. Visualize predictions vs actual values
4. Identify best performing model
5. Generate comparison plots and tables
6. Document findings in report

## Summary

All inference is configured for:
- ✅ Firozabad city only
- ✅ Nov 9-11, 2025 test period (72 hours)
- ✅ All 8 models comparison
- ✅ Comprehensive metrics
- ✅ Modal volume: `ai_ml_firozabad`
- ✅ H100 GPU optimization
