# Complete PM2.5 Prediction Pipeline - Firozabad
## From Data Collection to Visualization

---

## 📋 Overview

This guide covers the entire pipeline from scratch:
1. **Training Phase**: Collect historical data, preprocess, train 8 models
2. **Inference Phase**: Collect fresh Nov 9-11 data, run predictions
3. **Visualization Phase**: Analyze results and compare models

**Total Time**: ~8-10 hours (mostly training on Modal H100)

---

# PHASE 1: TRAINING (Historical Data)

## Step 1: Collect Training Data

**Purpose**: Fetch historical air quality and weather data for Firozabad

```bash
python data_collection.py
```

**What it does**:
- Collects data from Jan 1, 2022 to Nov 4, 2025
- Fetches from Open-Meteo API:
  - Air quality: PM2.5, PM10, CO, NO2, SO2, O3, dust, AOD, UV
  - Weather: Temperature, humidity, wind, pressure, radiation, cloud cover
- Saves to: `raw_pm25_firozabad_data.csv`

**Expected output**:
```
✓ DATA COLLECTION COMPLETE
Total records: ~32,000 hourly records
Date range: 2022-01-01 to 2025-11-04
Features: 32 raw parameters
✓ File saved: raw_pm25_firozabad_data.csv
```

**Time**: ~3-5 minutes

---

## Step 2: Preprocess Training Data

**Purpose**: Feature engineering - create 150+ features from raw data

```bash
python preprocessing.py
```

**What it does**:
- Handles missing values (interpolation, forward/backward fill)
- Adds temporal features (hour, day, cyclical encoding)
- Adds interaction features (temp×humidity, wind×pm10)
- Adds industrial features (furnace hours, stagnation index, ventilation)
- Adds secondary pollutant features (ozone formation, aerosol potential)
- Adds atmospheric stability features (Richardson proxy, mixing depth)
- Adds lag features (1, 2, 3, 6, 12, 24 hours)
- Adds rolling features (mean, std, min, max for 3, 6, 12, 24 hour windows)
- Saves to: `cleaned_pm25_firozabad_data.csv`

**Expected output**:
```
✓ PREPROCESSING COMPLETE
Final shape: (31,XXX, 155)
Total features: 154 (+ 1 time column)
  Temporal: 10
  Lag: 6
  Rolling: 16
  Industrial: 4
  Secondary pollutants: 3
  Atmospheric stability: 4
  Raw features: 111
✓ Saved: cleaned_pm25_firozabad_data.csv
```

**Time**: ~2-3 minutes

---

## Step 3: Upload Data to Modal

**Purpose**: Move processed data to Modal cloud for training

```bash
# Upload cleaned data
modal volume put ai_ml_firozabad cleaned_pm25_firozabad_data.csv /

# Upload model architecture file
modal volume put ai_ml_firozabad model_arch.py /
```

**Verify upload**:
```bash
modal volume ls ai_ml_firozabad
```

Should show:
- `cleaned_pm25_firozabad_data.csv`
- `model_arch.py`

**Time**: ~10-30 seconds

---

## Step 4: Create Sequences on Modal

**Purpose**: Convert tabular data into time-series sequences with train/val/test split

```bash
modal run create_sequences.py
```

**What it does**:
- Loads cleaned data from Modal volume
- Creates sequences with 168-hour (7-day) lookback
- Splits data: 70% train, 15% validation, 15% test
- Scales features and targets using StandardScaler
- Saves to: `firozabad_sequences_hourly_complete.pkl` on Modal volume

**Expected output**:
```
Processing firozabad...
  Loaded 31,XXX records
  Dropped non-feature columns
  Created sequences with 168-hour lookback
  Train: 21,XXX sequences
  Val: 4,XXX sequences
  Test: 4,XXX sequences
  Scaled features and targets
✓ Saved: firozabad_sequences_hourly_complete.pkl
```

**Time**: ~5-10 minutes

---

## Step 5: Train All 8 Models on Modal H100

**Purpose**: Train RandomForest, XGBoost, LSTM, GRU, BiLSTM, BiGRU, Enhanced_BiLSTM, Enhanced_BiGRU

```bash
modal run modal_training.py
```

**What it does**:
- Loads sequences from Modal volume
- **Phase 1 - ML Models** (~30 min total):
  - RandomForest: 200 trees, max_depth=20
  - XGBoost: GPU-accelerated, 1000 rounds
- **Phase 2 - Deep Learning Models** (~6-7 hours total):
  - Standard LSTM, GRU, BiLSTM, BiGRU: 256 hidden units, 3 layers
  - Enhanced BiLSTM, BiGRU: + Multi-head attention (8 heads) + Physics layer
  - Training: Max 150 epochs, early stopping patience=15, batch_size=128-256
  - Optimizations: H100 GPU, TF32, fused AdamW, gradient clipping
- Saves models to: `checkpoints_firozabad/` on Modal volume

**Expected output**:
```
PHASE 1: MACHINE LEARNING MODELS
================================
Training RandomForest...
  ✓ Epoch complete: Train RMSE=5.23, Val RMSE=6.45
  ✓ Saved: firozabad_RandomForest_model.joblib

Training XGBoost...
  ✓ Best iteration: 456, Val RMSE=6.12
  ✓ Saved: firozabad_XGBoost_model.joblib

PHASE 2: DEEP LEARNING MODELS
=============================
Training LSTM...
  Epoch 1/150: Train Loss=0.0234, Val Loss=0.0298, RMSE=7.23
  Epoch 2/150: Train Loss=0.0198, Val Loss=0.0276, RMSE=6.98
  ...
  Epoch 47/150: Early stopping triggered
  ✓ Best model saved at epoch 32: Val RMSE=6.54
  ✓ Saved: firozabad_LSTM_best.pth

[... similar for GRU, BiLSTM, BiGRU, Enhanced_BiLSTM, Enhanced_BiGRU ...]

✓ ALL MODELS TRAINED SUCCESSFULLY
Total time: ~7 hours
```

**Time**: ~7-8 hours total
- ML models: ~30 minutes
- Each DL model: ~45-60 minutes

**Cost**: ~$15-25 on Modal (H100 GPU time)

---

## Step 6: Verify Training Completed

```bash
# Check saved models
modal volume ls ai_ml_firozabad/checkpoints_firozabad/
```

Should show:
- `firozabad_RandomForest_model.joblib`
- `firozabad_XGBoost_model.joblib`
- `firozabad_LSTM_best.pth`
- `firozabad_GRU_best.pth`
- `firozabad_BiLSTM_best.pth`
- `firozabad_BiGRU_best.pth`
- `firozabad_Enhanced_BiLSTM_best.pth`
- `firozabad_Enhanced_BiGRU_best.pth`

✅ **Training Phase Complete!** Models are ready for inference.

---

# PHASE 2: INFERENCE (Fresh Nov 9-11, 2025 Data)

## Step 7: Collect Inference Data

**Purpose**: Fetch fresh data for Nov 2-11, 2025 (includes 168-hour lookback)

```bash
python collect_inference_data.py
```

**What it does**:
- Collects Nov 2-11, 2025 data (240 hours)
  - Nov 2-8: 168 hours for sequence lookback
  - Nov 9-11: 72 hours for actual predictions
- Uses same parameters as training data
- Saves to: `raw_pm25_firozabad_inference_nov9_11.csv`

**Expected output**:
```
INFERENCE DATA COLLECTION - NOV 2-11, 2025
=========================================
✓ Successfully fetched 240 hourly records
Date range: 2025-11-02 to 2025-11-11
Features collected: 32
PM2.5 data: Available: 235 / 240 hours
✓ File saved: raw_pm25_firozabad_inference_nov9_11.csv
```

**Time**: ~2-3 minutes

---

## Step 8: Preprocess Inference Data

**Purpose**: Apply same feature engineering as training

```bash
python preprocess_inference_data.py
```

**What it does**:
- Same preprocessing pipeline as training
- Creates same 154 features
- Handles missing values
- Saves to: `cleaned_pm25_firozabad_inference_nov9_11.csv`

**Expected output**:
```
INFERENCE DATA PREPROCESSING - NOV 2-11, 2025
============================================
Final shape: (240, 155)
Total features: 154
  Temporal: 10, Lag: 6, Rolling: 16
  Industrial: 4, Secondary: 3, Stability: 4
  Raw: 111
✓ Saved: cleaned_pm25_firozabad_inference_nov9_11.csv
```

**Time**: ~30 seconds

---

## Step 9: Upload Inference Data to Modal

**Purpose**: Make inference data accessible to Modal functions

```bash
modal volume put ai_ml_firozabad cleaned_pm25_firozabad_inference_nov9_11.csv /
```

**Verify**:
```bash
modal volume ls ai_ml_firozabad
```

Should now show:
- `cleaned_pm25_firozabad_data.csv` (training)
- `cleaned_pm25_firozabad_inference_nov9_11.csv` (inference) ✓
- `firozabad_sequences_hourly_complete.pkl` (training scalers) ✓
- `model_arch.py`
- `checkpoints_firozabad/` (trained models)

**Time**: ~5-10 seconds

---

## Step 10: Run Inference on All 8 Models

**Purpose**: Generate predictions for Nov 9-11 using trained models

```bash
modal run modal_inferencing.py
```

**What it does**:
- Loads fresh inference data (240 hours)
- Loads training scalers (for proper scaling)
- Creates 72 sequences (for Nov 9-11 predictions)
- For each of 8 models:
  - Loads trained model checkpoint
  - Scales data using training scalers
  - Generates predictions
  - Calculates metrics (RMSE, MAE, R², MAPE)
- Saves results to Modal volume

**Expected output**:
```
FIROZABAD PM2.5 INFERENCE - NOV 9-11, 2025 TEST SET
===================================================

PHASE 1: MACHINE LEARNING MODELS INFERENCE
==========================================

Running RandomForest...
  Loaded 240 inference records
  Loaded scalers from training data
  Created 72 sequences
  ✓ RMSE: 7.4523, MAE: 5.6234, R²: 0.8456

Running XGBoost...
  ✓ RMSE: 6.9234, MAE: 5.2123, R²: 0.8678

PHASE 2: DEEP LEARNING MODELS INFERENCE
=======================================

Running LSTM...
  ✓ RMSE: 7.1234, MAE: 5.4567, R²: 0.8543

Running GRU...
  ✓ RMSE: 7.0123, MAE: 5.3456, R²: 0.8598

Running BiLSTM...
  ✓ RMSE: 6.7891, MAE: 5.1234, R²: 0.8712

Running BiGRU...
  ✓ RMSE: 6.7234, MAE: 5.0987, R²: 0.8734

Running Enhanced_BiLSTM...
  ✓ RMSE: 6.2345, MAE: 4.7123, R²: 0.8923

Running Enhanced_BiGRU...
  ✓ RMSE: 6.1987, MAE: 4.6789, R²: 0.8945

INFERENCE COMPLETE - RESULTS SUMMARY
===================================

Model                     RMSE         MAE          R²         Status
---------------------------------------------------------------------------
RandomForest             7.4523       5.6234       0.8456     ✓
XGBoost                  6.9234       5.2123       0.8678     ✓
LSTM                     7.1234       5.4567       0.8543     ✓
GRU                      7.0123       5.3456       0.8598     ✓
BiLSTM                   6.7891       5.1234       0.8712     ✓
BiGRU                    6.7234       5.0987       0.8734     ✓
Enhanced_BiLSTM          6.2345       4.7123       0.8923     ✓
Enhanced_BiGRU           6.1987       4.6789       0.8945     ✓

✓ Results saved to Modal volume
```

**Time**: ~20-25 minutes

---

## Step 11: Download Inference Results

**Purpose**: Get results locally for visualization

```bash
# Download all results
modal volume get ai_ml_firozabad inference_results_firozabad/ .

# OR download just the JSON file
modal volume get ai_ml_firozabad inference_results_firozabad/firozabad_inference_nov9_11_2025.json .
```

**Downloaded files**:
- `firozabad_inference_nov9_11_2025.json` - Full predictions, targets, timestamps, metrics
- `metrics_summary_nov9_11_2025.json` - Just metrics

**Time**: ~5 seconds

---

# PHASE 3: VISUALIZATION & ANALYSIS

## Step 12: Generate Visualizations

**Purpose**: Create comprehensive plots and analysis

```bash
python visualization.py
```

**What it does**:
- Loads inference results JSON
- Creates for each model (8 files):
  - Time series plot (actual vs predicted)
  - Detailed hourly view
  - Scatter plot with R² score
  - Diurnal cycle (24-hour pattern)
  - Day vs Night comparison
  - Metrics table
- Creates comparison plots:
  - Model comparison (RMSE, MAE, R², MAPE bar charts)
  - Variance analysis (actual vs predicted)
- Generates performance summary CSV
- Identifies best-performing models

**Expected output**:
```
STEP 7: FIROZABAD PM2.5 VISUALIZATION & ANALYSIS (NOV 9-11, 2025)
=================================================================

Creating visualizations for Firozabad...

✓ Saved: results_firozabad/firozabad_RandomForest_nov9_11_analysis.png
✓ Saved: results_firozabad/firozabad_XGBoost_nov9_11_analysis.png
✓ Saved: results_firozabad/firozabad_LSTM_nov9_11_analysis.png
✓ Saved: results_firozabad/firozabad_GRU_nov9_11_analysis.png
✓ Saved: results_firozabad/firozabad_BiLSTM_nov9_11_analysis.png
✓ Saved: results_firozabad/firozabad_BiGRU_nov9_11_analysis.png
✓ Saved: results_firozabad/firozabad_Enhanced_BiLSTM_nov9_11_analysis.png
✓ Saved: results_firozabad/firozabad_Enhanced_BiGRU_nov9_11_analysis.png

Creating comparison visualizations...
✓ Saved: results_firozabad/model_comparison_nov9_11.png
✓ Saved: results_firozabad/variance_comparison_nov9_11.png

FIROZABAD PM2.5 PREDICTION - PERFORMANCE SUMMARY (NOV 9-11, 2025)
=================================================================

Model                    RMSE      MAE       R2       MAPE    Variance_Diff  Num_Samples
RandomForest            7.4523    5.6234    0.8456   14.23      28.45           72
XGBoost                 6.9234    5.2123    0.8678   13.12      24.56           72
LSTM                    7.1234    5.4567    0.8543   13.67      26.34           72
GRU                     7.0123    5.3456    0.8598   13.45      25.78           72
BiLSTM                  6.7891    5.1234    0.8712   12.89      23.12           72
BiGRU                   6.7234    5.0987    0.8734   12.67      22.89           72
Enhanced_BiLSTM         6.2345    4.7123    0.8923   11.45      19.67           72
Enhanced_BiGRU          6.1987    4.6789    0.8945   11.23      19.34           72

✓ Saved: results_firozabad/performance_summary_nov9_11.csv

BEST PERFORMING MODELS
=====================
Lowest RMSE: Enhanced_BiGRU (RMSE=6.1987, R²=0.8945)
Highest R²:  Enhanced_BiGRU (R²=0.8945, RMSE=6.1987)

✓ VISUALIZATION COMPLETE

Generated files in 'results_firozabad/' folder:
  • firozabad_{model}_nov9_11_analysis.png - Individual model analysis (8 files)
  • model_comparison_nov9_11.png - Metrics comparison across all models
  • variance_comparison_nov9_11.png - Variance analysis
  • performance_summary_nov9_11.csv - Complete metrics table
```

**Generated files** (in `results_firozabad/`):
- 8 individual model analysis PNGs
- 1 model comparison PNG
- 1 variance comparison PNG
- 1 performance summary CSV

**Time**: ~1-2 minutes

---

# 📊 Summary of Complete Pipeline

## Commands in Order

```bash
# ===== PHASE 1: TRAINING =====
python data_collection.py                    # Step 1: Collect historical data
python preprocessing.py                      # Step 2: Feature engineering
modal volume put ai_ml_firozabad cleaned_pm25_firozabad_data.csv /  # Step 3
modal volume put ai_ml_firozabad model_arch.py /
modal run create_sequences.py                # Step 4: Create sequences with train/val/test split
modal run modal_training.py                  # Step 5: Train 8 models (~7 hours)

# ===== PHASE 2: INFERENCE =====
python collect_inference_data.py             # Step 7: Collect Nov 2-11 data
python preprocess_inference_data.py          # Step 8: Preprocess
modal volume put ai_ml_firozabad cleaned_pm25_firozabad_inference_nov9_11.csv /  # Step 9
modal run modal_inferencing.py               # Step 10: Run inference on all models
modal volume get ai_ml_firozabad inference_results_firozabad/ .  # Step 11

# ===== PHASE 3: VISUALIZATION =====
python visualization.py                      # Step 12: Generate plots and analysis
```

## Timeline

| Phase | Steps | Time |
|-------|-------|------|
| **Training** | 1-6 | ~8-9 hours |
| **Inference** | 7-11 | ~30 minutes |
| **Visualization** | 12 | ~2 minutes |
| **Total** | 1-12 | **~9-10 hours** |

*Note: Training time is mostly unattended GPU time on Modal*

## File Flow

```
TRAINING:
raw_pm25_firozabad_data.csv (local)
  ↓
cleaned_pm25_firozabad_data.csv (local → Modal)
  ↓
firozabad_sequences_hourly_complete.pkl (Modal)
  ↓
checkpoints_firozabad/*.joblib/*.pth (Modal)

INFERENCE:
raw_pm25_firozabad_inference_nov9_11.csv (local)
  ↓
cleaned_pm25_firozabad_inference_nov9_11.csv (local → Modal)
  ↓ (uses trained models + scalers)
inference_results_firozabad/*.json (Modal → local)

VISUALIZATION:
firozabad_inference_nov9_11_2025.json (local)
  ↓
results_firozabad/*.png, *.csv (local)
```

## Data Splits Explained

### Training Phase (Steps 1-6)
- **Historical data**: Jan 2022 - Nov 4, 2025
- **Split**: 70% train / 15% validation / 15% test
- **Purpose**: Train models, tune hyperparameters, evaluate on held-out test set

### Inference Phase (Steps 7-11)
- **Fresh data**: Nov 2-11, 2025 (NOT in training data)
- **No split**: All 72 hours (Nov 9-11) are predicted
- **Purpose**: Real-world testing on completely unseen future data

## Cost Breakdown (Modal)

| Component | Time | Approx Cost |
|-----------|------|-------------|
| Create sequences | 10 min | ~$0.20 |
| Train ML models | 30 min | ~$0.60 |
| Train DL models | 7 hours | ~$14-20 |
| Run inference | 25 min | ~$0.50 |
| **Total** | **~8 hours** | **~$15-25** |

*Prices based on Modal H100 GPU rates*

## Success Checklist

### After Training (Step 6)
- [ ] 8 model files exist in `checkpoints_firozabad/`
- [ ] Training completed without errors
- [ ] Validation metrics look reasonable (R² > 0.7)

### After Inference (Step 11)
- [ ] All 8 models ran successfully
- [ ] Results JSON downloaded
- [ ] Metrics are reasonable (R² > 0.5, RMSE < 20)

### After Visualization (Step 12)
- [ ] 10 PNG files + 1 CSV generated
- [ ] Best model identified (usually Enhanced_BiGRU or Enhanced_BiLSTM)
- [ ] Plots show sensible patterns (not random noise)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API timeout in Step 1/7 | Retry, check internet connection |
| Missing values in Step 2/8 | Normal, handled by interpolation |
| Modal upload fails in Step 3/9 | Check Modal authentication |
| Training crashes in Step 5 | Check Modal credits, reduce batch size |
| Inference fails in Step 10 | Verify all files uploaded to Modal |
| Poor metrics (R² < 0) | Check scalers, feature alignment |

## Next Steps After Pipeline Completion

1. **Analyze Results**: Check which model performed best
2. **Error Analysis**: Look at hours with high prediction errors
3. **Deploy Best Model**: Use for real-time predictions
4. **Monitor Performance**: Track accuracy over time
5. **Retrain**: When performance degrades or new data available

---

## Quick Command Reference

```bash
# Check Modal volume contents
modal volume ls ai_ml_firozabad

# Check specific folder
modal volume ls ai_ml_firozabad/checkpoints_firozabad/

# Download specific file
modal volume get ai_ml_firozabad <filename> .

# View Modal logs
modal logs ai_ml_firozabad

# Check Modal app runs
modal app list
```

---

🎉 **Pipeline Complete!** You now have trained models and predictions for Nov 9-11, 2025.
