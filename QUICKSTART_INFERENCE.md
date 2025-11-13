# Quick Start: Inference on Nov 9-11, 2025

## One-Line Summary
Collect fresh Nov 2-11 data → Preprocess → Upload to Modal → Run inference → Visualize

---

## Commands in Order

```bash
# 1. Collect fresh data (Nov 2-11, 2025 = 240 hours)
python collect_inference_data.py

# 2. Preprocess with same features as training
python preprocess_inference_data.py

# 3. Upload to Modal volume
modal volume put ai_ml_firozabad cleaned_pm25_firozabad_inference_nov9_11.csv /

# 4. Run inference on all 8 models (on Modal H100)
modal run modal_inferencing.py

# 5. Download results
modal volume get ai_ml_firozabad inference_results_firozabad/ .

# 6. Create visualizations
python visualization.py
```

---

## Why Nov 2-11 (not just Nov 9-11)?

**Need 168-hour lookback for sequences:**
- Nov 2-8: 168 hours = lookback data
- Nov 9-11: 72 hours = actual predictions

**Formula:** Prediction_hours + Lookback_hours = Total_hours
- 72 + 168 = 240 hours
- 240 / 24 = 10 days (Nov 2-11)

---

## Critical Points

1. **Use Training Scalers** (not new scalers)
2. **Same Features** (150+ features from preprocessing)
3. **Same Sequence Length** (168 hours lookback)
4. **Fresh Data** (not from training set)

---

## Expected Timeline

| Step | Time | Output |
|------|------|--------|
| Data Collection | 2-3 min | 240 hours of raw data |
| Preprocessing | 30 sec | 240 rows × 155 features |
| Upload to Modal | 10 sec | CSV on cloud |
| Inference (8 models) | 20-25 min | Predictions + metrics |
| Download | 5 sec | JSON results |
| Visualization | 1 min | PNG plots + CSV table |
| **Total** | **~30 min** | Complete analysis |

---

## Success Indicators

✅ **Data Collection**: 240 hours collected  
✅ **Preprocessing**: 154 features created  
✅ **Sequences**: 72 sequences (Nov 9-11)  
✅ **Models**: All 8 completed  
✅ **Metrics**: R² > 0.5, RMSE < 20  
✅ **Visualizations**: 13 files in results_firozabad/  

---

## Files Generated

```
Local:
├── raw_pm25_firozabad_inference_nov9_11.csv          (240 hours raw)
├── cleaned_pm25_firozabad_inference_nov9_11.csv      (240 hours processed)
├── firozabad_inference_nov9_11_2025.json             (predictions + metrics)
├── metrics_summary_nov9_11_2025.json                 (just metrics)
└── results_firozabad/
    ├── firozabad_RandomForest_nov9_11_analysis.png   (8 model plots)
    ├── firozabad_XGBoost_nov9_11_analysis.png
    ├── ... (6 more)
    ├── model_comparison_nov9_11.png                   (comparison)
    ├── variance_comparison_nov9_11.png                (variance)
    └── performance_summary_nov9_11.csv                (metrics table)

Modal Volume (ai_ml_firozabad):
├── cleaned_pm25_firozabad_inference_nov9_11.csv      (uploaded)
└── inference_results_firozabad/
    ├── firozabad_inference_nov9_11_2025.json
    └── metrics_summary_nov9_11_2025.json
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| "Not enough data" | Collect Nov 2-11 (not just Nov 9-11) |
| "Model not found" | Complete training first |
| "Feature mismatch" | Check preprocessing matches training |
| "Negative R²" | Check scalers (must use training scalers) |

---

## Quick Validation

After each step, verify:

```bash
# After Step 1
wc -l raw_pm25_firozabad_inference_nov9_11.csv
# Should show: 241 lines (240 data + 1 header)

# After Step 2
wc -l cleaned_pm25_firozabad_inference_nov9_11.csv
# Should show: 241 lines

# After Step 3
modal volume ls ai_ml_firozabad
# Should show: cleaned_pm25_firozabad_inference_nov9_11.csv

# After Step 4
modal volume ls ai_ml_firozabad/inference_results_firozabad/
# Should show: 2 JSON files

# After Step 6
ls results_firozabad/*.png | wc -l
# Should show: 10 (8 model plots + 2 comparison plots)
```

---

## Best Model Selection

After visualization, check `performance_summary_nov9_11.csv`:

```csv
Model,RMSE,MAE,R2,MAPE,Variance_Diff,Num_Samples
Enhanced_BiLSTM,7.234,5.123,0.891,12.4,23.1,72
Enhanced_BiGRU,7.456,5.234,0.885,13.1,24.5,72
XGBoost,8.123,6.012,0.867,14.8,31.2,72
...
```

**Pick model with:**
- Lowest RMSE ✓
- Highest R² ✓
- Lowest MAPE ✓

Usually: **Enhanced_BiLSTM** or **Enhanced_BiGRU** wins

---

## Done! 🎉

You now have:
- ✅ Predictions for Nov 9-11, 2025 (72 hours)
- ✅ Performance metrics for 8 models
- ✅ Comprehensive visualizations
- ✅ Best model identified

Use the best model for deployment or further analysis.
