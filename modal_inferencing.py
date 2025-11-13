"""
Step 6: Inference on Modal H100 GPU - Firozabad Nov 9-11, 2025
- Uses fresh data collected for Nov 9-11, 2025
- Loads trained models and scalers from training
- Runs inference for all 8 models
- Saves results to Modal volume
"""
import modal
from pathlib import Path
import numpy as np
import torch
import json

volume = modal.Volume.from_name("ai_ml_firozabad", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "pandas==2.2.0",
        "numpy==1.26.0",
        "scikit-learn==1.4.0",
        "xgboost==2.0.3",
        "joblib==1.3.2"
    )
)

app = modal.App("pm25-firozabad-inference", image=image)

VOLUME_PATH = Path("/data")
RESULTS_DIR = VOLUME_PATH / "inference_results_firozabad"
LOOKBACK = 168  # 7 days = 168 hours (must match training)

@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/data": volume},
    cpu=8.0,
    memory=32768
)
def run_ml_inference(city_name, model_type='RandomForest'):
    """Run inference for ML models (RandomForest, XGBoost) using fresh Nov 9-11 data"""
    
    import pickle
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    print(f"Running inference for {city_name} - {model_type}...")
    
    # Load fresh inference data
    inference_file = VOLUME_PATH / 'cleaned_pm25_firozabad_inference_nov9_11.csv'
    
    try:
        df_inference = pd.read_csv(inference_file)
        print(f"  Loaded {len(df_inference)} inference records")
    except FileNotFoundError:
        print(f"ERROR: Inference data not found at {inference_file}")
        print("Please upload cleaned_pm25_firozabad_inference_nov9_11.csv to Modal volume first")
        return None
    
    # Load training scalers
    scaler_file = VOLUME_PATH / f"{city_name}_sequences_hourly_complete.pkl"
    try:
        with open(scaler_file, 'rb') as f:
            train_data = pickle.load(f)
        scaler_features = train_data['scaler_features']
        scaler_target = train_data['scaler_target']
        print(f"  Loaded scalers from training data")
    except FileNotFoundError:
        print(f"ERROR: Training scalers not found at {scaler_file}")
        return None
    
    # Prepare features - MUST MATCH create_sequences.py exclusion list!
    # NOTE: pm2_5 is INCLUDED in features for scaling, then used separately as target
    exclude_cols = ['time', 'city', 'latitude', 'longitude', 'hour', 'day_of_week',
                    'month', 'day_of_year', 'week_of_year', 'diurnal_period', 'hour_group']
    feature_cols = [c for c in df_inference.columns if c not in exclude_cols]
    target_col = 'pm2_5'
    
    # Extract features and target
    X_inference = df_inference[feature_cols].values
    y_inference = df_inference[target_col].values
    time_index = pd.to_datetime(df_inference['time'])
    
    print(f"  Features shape: {X_inference.shape}")
    print(f"  Need {LOOKBACK} hours lookback for sequences")
    
    # Create sequences for inference
    sequences = []
    targets = []
    valid_times = []
    
    for i in range(LOOKBACK, len(X_inference)):
        sequences.append(X_inference[i-LOOKBACK:i])
        targets.append(y_inference[i])
        valid_times.append(time_index.iloc[i])
    
    if len(sequences) == 0:
        print(f"ERROR: Not enough data for sequences. Need at least {LOOKBACK} hours")
        return None
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    print(f"  Created {len(sequences)} sequences")
    print(f"  Sequence shape: {sequences.shape}")
    
    # Scale features and targets
    sequences_scaled = scaler_features.transform(sequences.reshape(-1, sequences.shape[-1]))
    sequences_scaled = sequences_scaled.reshape(sequences.shape)
    targets_scaled = scaler_target.transform(targets.reshape(-1, 1))
    
    # Flatten for ML models
    X_test = sequences_scaled.reshape(len(sequences), -1)
    y_test = targets_scaled.flatten()
    
    print(f"  Flattened sequence shape: {X_test.shape}")
    
    # Load feature selector if it exists (used during training to reduce dimensionality)
    selector_path = VOLUME_PATH / "checkpoints_firozabad" / f"{city_name}_feature_selector.joblib"
    try:
        selector = joblib.load(selector_path)
        print(f"  Loaded feature selector from {selector_path}")
        X_test = selector.transform(X_test)
        print(f"  Applied feature selection: {X_test.shape}")
    except FileNotFoundError:
        print(f"  No feature selector found (model trained on full features)")
    
    # Load model
    model_path = VOLUME_PATH / "checkpoints_firozabad" / f"{city_name}_{model_type}_model.joblib"
    
    try:
        model = joblib.load(model_path)
        print(f"  Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model not found at {model_path}")
        return None
    
    # Predict
    predictions = model.predict(X_test)
    
    # Inverse transform
    predictions_orig = scaler_target.inverse_transform(predictions.reshape(-1, 1))
    targets_orig = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    
    # Metrics
    mse = mean_squared_error(targets_orig, predictions_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_orig, predictions_orig)
    r2 = r2_score(targets_orig, predictions_orig)
    mape = np.mean(np.abs((targets_orig - predictions_orig) / (targets_orig + 1e-8))) * 100
    
    variance_actual = np.var(targets_orig)
    variance_pred = np.var(predictions_orig)
    variance_diff = np.abs(variance_actual - variance_pred)
    
    time_index_list = [t.strftime('%Y-%m-%d %H:%M:%S') for t in valid_times]
    
    result = {
        'city': city_name,
        'model_type': model_type,
        'test_period': 'Nov 9-11, 2025',
        'num_samples': len(sequences),
        'predictions': predictions_orig.flatten().tolist(),
        'targets': targets_orig.flatten().tolist(),
        'time_index': time_index_list,
        'metrics': {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'MAPE': float(mape),
            'Variance_Actual': float(variance_actual),
            'Variance_Pred': float(variance_pred),
            'Variance_Diff': float(variance_diff)
        }
    }
    
    print(f"✓ Inference complete: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    return result


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/data": volume},
    cpu=8.0,
    memory=32768
)
def run_deep_inference(city_name, model_type='LSTM'):
    """Run inference for deep learning models using fresh Nov 9-11 data"""
    
    import torch
    import pickle
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import sys
    
    sys.path.append(str(VOLUME_PATH))
    from model_arch import (
        StandardLSTM, StandardGRU, StandardBiLSTM, StandardBiGRU,
        EnhancedBiLSTMAttentionComplete, EnhancedBiGRUAttentionComplete
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Running inference for {city_name} - {model_type}...")
    
    # Load fresh inference data
    inference_file = VOLUME_PATH / 'cleaned_pm25_firozabad_inference_nov9_11.csv'
    
    try:
        df_inference = pd.read_csv(inference_file)
        print(f"  Loaded {len(df_inference)} inference records")
    except FileNotFoundError:
        print(f"ERROR: Inference data not found at {inference_file}")
        print("Please upload cleaned_pm25_firozabad_inference_nov9_11.csv to Modal volume first")
        return None
    
    # Load training scalers
    scaler_file = VOLUME_PATH / f"{city_name}_sequences_hourly_complete.pkl"
    try:
        with open(scaler_file, 'rb') as f:
            train_data = pickle.load(f)
        scaler_features = train_data['scaler_features']
        scaler_target = train_data['scaler_target']
        print(f"  Loaded scalers from training data")
    except FileNotFoundError:
        print(f"ERROR: Training scalers not found at {scaler_file}")
        return None
    
    # Prepare features - MUST MATCH create_sequences.py exclusion list!
    # NOTE: pm2_5 is INCLUDED in features for scaling, then used separately as target
    exclude_cols = ['time', 'city', 'latitude', 'longitude', 'hour', 'day_of_week',
                    'month', 'day_of_year', 'week_of_year', 'diurnal_period', 'hour_group']
    feature_cols = [c for c in df_inference.columns if c not in exclude_cols]
    target_col = 'pm2_5'
    
    # Extract features and target
    X_inference = df_inference[feature_cols].values
    y_inference = df_inference[target_col].values
    time_index = pd.to_datetime(df_inference['time'])
    
    print(f"  Features shape: {X_inference.shape}")
    print(f"  Need {LOOKBACK} hours lookback for sequences")
    
    # Create sequences for inference
    sequences = []
    targets = []
    valid_times = []
    
    for i in range(LOOKBACK, len(X_inference)):
        sequences.append(X_inference[i-LOOKBACK:i])
        targets.append(y_inference[i])
        valid_times.append(time_index.iloc[i])
    
    if len(sequences) == 0:
        print(f"ERROR: Not enough data for sequences. Need at least {LOOKBACK} hours")
        print(f"Available: {len(X_inference)} hours")
        print(f"NOTE: You need 168 hours of historical data BEFORE Nov 9 to predict Nov 9-11")
        return None
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    print(f"  Created {len(sequences)} sequences")
    print(f"  Sequence shape: {sequences.shape}")
    
    # Scale features and targets
    sequences_scaled = scaler_features.transform(sequences.reshape(-1, sequences.shape[-1]))
    sequences_scaled = sequences_scaled.reshape(sequences.shape)
    targets_scaled = scaler_target.transform(targets.reshape(-1, 1))
    
    # Load model
    input_size = sequences_scaled.shape[2]
    
    # Map model types to classes
    model_classes = {
        'LSTM': StandardLSTM,
        'GRU': StandardGRU,
        'BiLSTM': StandardBiLSTM,
        'BiGRU': StandardBiGRU,
        'Enhanced_BiLSTM': EnhancedBiLSTMAttentionComplete,
        'Enhanced_BiGRU': EnhancedBiGRUAttentionComplete
    }
    
    if model_type not in model_classes:
        print(f"ERROR: Unknown model type {model_type}")
        return None
    
    model_class = model_classes[model_type]
    
    # Standard models have different init signature
    if 'Enhanced' in model_type:
        model = model_class(input_size, 256, 3, 0.3, 8)
    else:
        model = model_class(input_size, 256, 3, 0.3)
    
    checkpoint_path = VOLUME_PATH / "checkpoints_firozabad" / f"{city_name}_{model_type}_best.pth"
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"  Loaded model from {checkpoint_path}")
    except FileNotFoundError:
        print(f"ERROR: Model checkpoint not found at {checkpoint_path}")
        print("Make sure training was completed first")
        return None
    
    model = model.to(device)
    model.eval()
    
    # Convert to tensors
    test_sequences = torch.FloatTensor(sequences_scaled).to(device)
    test_targets = targets_scaled
    
    print(f"  Running inference on {len(test_sequences)} sequences...")
    
    predictions = []
    batch_size = 128
    
    with torch.no_grad():
        for i in range(0, len(test_sequences), batch_size):
            batch = test_sequences[i:i+batch_size]
            
            # Enhanced models need physics features
            if 'Enhanced' in model_type:
                last_ts = batch[:, -1, :]
                physics_features = {
                    'is_night': last_ts[:, -23:-22],
                    'hour_norm': (last_ts[:, 0:1] / 24.0),
                    'wind': last_ts[:, -15:-13],
                    'pbl_proxy': last_ts[:, -22:-21],
                    'humidity': last_ts[:, -20:-19],
                    'temperature': last_ts[:, -21:-20]
                }
                outputs, _ = model(batch, physics_features)
            else:
                # Standard models also return tuple (outputs, attention_weights)
                outputs, _ = model(batch)
            
            predictions.append(outputs.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Inverse transform
    predictions_orig = scaler_target.inverse_transform(predictions)
    targets_orig = scaler_target.inverse_transform(test_targets)
    
    # Metrics
    mse = mean_squared_error(targets_orig, predictions_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_orig, predictions_orig)
    r2 = r2_score(targets_orig, predictions_orig)
    mape = np.mean(np.abs((targets_orig - predictions_orig) / (targets_orig + 1e-8))) * 100
    
    variance_actual = np.var(targets_orig)
    variance_pred = np.var(predictions_orig)
    variance_diff = np.abs(variance_actual - variance_pred)
    
    # Convert time_index to strings
    time_index_list = [t.strftime('%Y-%m-%d %H:%M:%S') for t in valid_times]
    
    result = {
        'city': city_name,
        'model_type': model_type,
        'test_period': 'Nov 9-11, 2025',
        'num_samples': len(sequences),
        'predictions': predictions_orig.flatten().tolist(),
        'targets': targets_orig.flatten().tolist(),
        'time_index': time_index_list,
        'metrics': {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'MAPE': float(mape),
            'Variance_Actual': float(variance_actual),
            'Variance_Pred': float(variance_pred),
            'Variance_Diff': float(variance_diff)
        }
    }
    
    print(f"✓ Inference complete: RMSE={rmse:.4f}, R²={r2:.4f}")
    
    return result

@app.function(
    timeout=1800,
    volumes={"/data": volume}
)
def save_results_to_volume(results_dict):
    """Save results to Modal volume as JSON"""
    
    import json
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save complete results
    results_file = RESULTS_DIR / 'firozabad_inference_nov9_11_2025.json'
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"✓ Results saved to Modal volume: {results_file}")
    
    # Also save metrics summary
    summary = {}
    for model_type, result in results_dict.items():
        if result is not None:
            summary[model_type] = result['metrics']
    
    summary_file = RESULTS_DIR / 'metrics_summary_nov9_11_2025.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Metrics summary saved: {summary_file}")
    
    return str(results_file)

@app.local_entrypoint()
def main():
    """Run inference for all 8 models on Firozabad Nov 9-11, 2025 data"""
    
    city = 'firozabad'
    ml_models = ['RandomForest', 'XGBoost']
    deep_models = ['LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'Enhanced_BiLSTM', 'Enhanced_BiGRU']
    
    print("\n" + "="*80)
    print("FIROZABAD PM2.5 INFERENCE - NOV 9-11, 2025 TEST SET")
    print("="*80)
    print(f"\nCity: {city.upper()}")
    print(f"Test Period: November 9-11, 2025 (72 hours)")
    print(f"Total Models: 8")
    print(f"  ML Models: {', '.join(ml_models)}")
    print(f"  Deep Learning Models: {', '.join(deep_models)}")
    print(f"\nNOTE: This uses FRESH data collected for Nov 9-11, 2025")
    print(f"      Make sure 'cleaned_pm25_firozabad_inference_nov9_11.csv'")
    print(f"      has been uploaded to Modal volume 'ai_ml_firozabad'\n")
    
    results = {}
    
    # Phase 1: ML Models
    print("="*80)
    print("PHASE 1: MACHINE LEARNING MODELS INFERENCE")
    print("="*80 + "\n")
    
    for model_type in ml_models:
        print(f"Running {model_type}...")
        result = run_ml_inference.remote(city, model_type)
        
        if result is None:
            print(f"  ✗ ERROR: Inference failed for {model_type}\n")
            results[model_type] = None
        else:
            results[model_type] = result
            print(f"  ✓ RMSE: {result['metrics']['RMSE']:.4f}, "
                  f"MAE: {result['metrics']['MAE']:.4f}, "
                  f"R²: {result['metrics']['R2']:.4f}\n")
    
    # Phase 2: Deep Learning Models
    print("="*80)
    print("PHASE 2: DEEP LEARNING MODELS INFERENCE")
    print("="*80 + "\n")
    
    for model_type in deep_models:
        print(f"Running {model_type}...")
        result = run_deep_inference.remote(city, model_type)
        
        if result is None:
            print(f"  ✗ ERROR: Inference failed for {model_type}\n")
            results[model_type] = None
        else:
            results[model_type] = result
            print(f"  ✓ RMSE: {result['metrics']['RMSE']:.4f}, "
                  f"MAE: {result['metrics']['MAE']:.4f}, "
                  f"R²: {result['metrics']['R2']:.4f}\n")
    
    # Save results
    print("="*80)
    print("SAVING RESULTS")
    print("="*80 + "\n")
    
    save_results_to_volume.remote(results)
    
    # Print summary
    print("\n" + "="*80)
    print("INFERENCE COMPLETE - RESULTS SUMMARY")
    print("="*80 + "\n")
    
    print(f"{'Model':<25} {'RMSE':<12} {'MAE':<12} {'R²':<10} {'Status'}")
    print("-"*75)
    
    for model_type in ml_models + deep_models:
        if model_type in results and results[model_type] is not None:
            metrics = results[model_type]['metrics']
            print(f"{model_type:<25} "
                  f"{metrics['RMSE']:<12.4f} "
                  f"{metrics['MAE']:<12.4f} "
                  f"{metrics['R2']:<10.4f} "
                  f"{'✓'}")
        else:
            print(f"{model_type:<25} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'✗ FAILED'}")
    
    print("\n" + "="*80)
    print("To download results locally, run:")
    print("  modal volume get ai_ml_firozabad inference_results_firozabad/ .")
    print("\nOr download just the results JSON:")
    print("  modal volume get ai_ml_firozabad inference_results_firozabad/firozabad_inference_nov9_11_2025.json .")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
