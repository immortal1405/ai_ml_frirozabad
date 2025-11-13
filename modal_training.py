"""
Step 5: Complete Training on Modal H100 GPU - 8 Model Comparison
Models: RandomForest, XGBoost, LSTM, GRU, BiLSTM, BiGRU, Enhanced BiLSTM+Attention+Physics, Enhanced BiGRU+Attention+Physics
Run locally with: modal run modal_training.py
"""
import modal
from pathlib import Path

# Create volume
volume = modal.Volume.from_name("ai_ml_firozabad", create_if_missing=True)

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "pandas==2.2.0",
        "numpy==1.26.0",
        "scikit-learn==1.4.0",
        "xgboost==2.0.3",
        "tqdm==4.66.0",
        "matplotlib==3.8.0",
        "joblib==1.3.2"
    )
)

app = modal.App("pm25-firozabad-training", image=image)

VOLUME_PATH = Path("/data")
CHECKPOINT_DIR = VOLUME_PATH / "checkpoints_firozabad"

@app.function(
    gpu="H100",
    timeout=86400,  # 24 hours
    volumes={"/data": volume},
    retries=modal.Retries(max_retries=1, initial_delay=10.0),
    cpu=16.0,  # More CPU for data loading
    memory=65536  # 64GB RAM for large datasets
)
def train_ml_model(city_name, model_type='RandomForest'):
    """Train traditional ML models (RandomForest, XGBoost)"""
    
    import pickle
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import xgboost as xgb
    import json
    from time import time
    import joblib
    
    print(f"\n{'='*80}")
    print(f"Training {model_type} for {city_name}")
    print(f"{'='*80}\n")
    
    # Load data
    data_file = VOLUME_PATH / f"{city_name}_sequences_hourly_complete.pkl"
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data loaded:")
    print(f"  Train: {data['train_seq'].shape}")
    print(f"  Val: {data['val_seq'].shape}")
    print(f"  Test: {data['test_seq'].shape}\n")
    
    # Flatten sequences for ML models
    X_train = data['train_seq'].reshape(data['train_seq'].shape[0], -1)
    y_train = data['train_target'].flatten()
    X_val = data['val_seq'].reshape(data['val_seq'].shape[0], -1)
    y_val = data['val_target'].flatten()
    X_test = data['test_seq'].reshape(data['test_seq'].shape[0], -1)
    y_test = data['test_target'].flatten()
    
    print(f"Flattened shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}\n")
    
    # Feature selection for ML models (reduce dimensionality)
    if X_train.shape[1] > 5000:
        print(f"⚠ High dimensionality detected: {X_train.shape[1]} features")
        print(f"Applying feature selection to reduce training time...\n")
        
        from sklearn.feature_selection import SelectKBest, f_regression
        
        # Select top 2000 features (good balance of performance and speed)
        k_features = min(2000, X_train.shape[1] // 2)
        print(f"Selecting top {k_features} features using f_regression...")
        
        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_train = selector.fit_transform(X_train, y_train)
        X_val = selector.transform(X_val)
        X_test = selector.transform(X_test)
        
        print(f"✓ Feature selection complete")
        print(f"  New X_train shape: {X_train.shape}")
        print(f"  This will speed up training significantly!\n")
        
        # Save selector for inference
        import joblib
        selector_path = CHECKPOINT_DIR / f"{city_name}_feature_selector.joblib"
        selector_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(selector, selector_path)
    
    # Initialize model
    start_time = time()
    
    if model_type == 'RandomForest':
        print(f"Initializing RandomForest...")
        # Reduced to 100 trees for faster training with high-dimensional data
        n_trees = 100 if X_train.shape[1] > 2000 else 200
        print(f"Using {n_trees} trees (optimized for {X_train.shape[1]:,} features)")
        print(f"Estimated time: 15-30 minutes\n")
        model = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=20,  # Reduced from 30
            min_samples_split=10,  # Increased from 5
            min_samples_leaf=4,  # Increased from 2
            max_features='sqrt',  # Only consider sqrt(n_features) per split - huge speedup!
            n_jobs=-1,
            random_state=42,
            verbose=2,
            warm_start=False
        )
    elif model_type == 'XGBoost':
        print(f"Initializing XGBoost...")
        print(f"Training on GPU with hist method\n")
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            tree_method='hist',
            device='cuda',
            verbosity=2  # More verbose
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Training {model_type}...")
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Features per sample: {X_train.shape[1]:,}")
    
    from datetime import datetime
    print(f"This will take some time. Training started at: {datetime.now().strftime('%H:%M:%S')}\n")
    
    # Add progress callback for scikit-learn models
    import sys
    from io import StringIO
    
    # Capture and print output in real-time
    model.fit(X_train, y_train)
    
    print(f"\n✓ Training completed at: {datetime.now().strftime('%H:%M:%S')}")
    training_time = time() - start_time
    print(f"✓ Training completed in {training_time:.2f} seconds\n")
    
    # Evaluate
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Validation Metrics:")
    print(f"  MSE: {val_mse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.6f}")
    print(f"Test Metrics:")
    print(f"  MSE: {test_mse:.6f}, MAE: {test_mae:.6f}, R²: {test_r2:.6f}\n")
    
    # Save model
    model_path = CHECKPOINT_DIR / f"{city_name}_{model_type}_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    
    # Save results
    results = {
        'city': city_name,
        'model_type': model_type,
        'val_mse': float(val_mse),
        'val_mae': float(val_mae),
        'val_r2': float(val_r2),
        'test_mse': float(test_mse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'training_time': training_time
    }
    
    results_file = CHECKPOINT_DIR / f"{city_name}_{model_type}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    volume.commit()
    
    print(f"{'='*80}")
    print(f"✓ Training complete for {city_name} - {model_type}")
    print(f"{'='*80}\n")
    
    return results


@app.function(
    gpu="H100",
    timeout=86400,  # 24 hours
    volumes={"/data": volume},
    retries=modal.Retries(max_retries=1, initial_delay=10.0),
    cpu=16.0,
    memory=65536
)
def train_deep_model(city_name, model_type='LSTM'):
    """Train deep learning models (LSTM, GRU, BiLSTM, BiGRU, Enhanced variants)"""
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import pickle
    import numpy as np
    from tqdm import tqdm
    import json
    import sys
    
    sys.path.append(str(VOLUME_PATH))
    from model_arch import (
        StandardLSTM, StandardGRU, StandardBiLSTM, StandardBiGRU,
        EnhancedBiLSTMAttentionComplete, EnhancedBiGRUAttentionComplete
    )
    
    print(f"\n{'='*80}")
    print(f"Training {model_type} for {city_name}")
    print(f"{'='*80}\n")
    
    # GPU setup with H100 optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
        
        # H100 optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')  # H100 specific
    
    # Load data
    data_file = VOLUME_PATH / f"{city_name}_sequences_hourly_complete.pkl"
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data loaded:")
    print(f"  Train: {data['train_seq'].shape}")
    print(f"  Val: {data['val_seq'].shape}")
    print(f"  Test: {data['test_seq'].shape}")
    print(f"  Sequence length: {data['sequence_length']} hours\n")
    
    # DataLoaders with optimized settings for H100
    train_dataset = TensorDataset(
        torch.FloatTensor(data['train_seq']),
        torch.FloatTensor(data['train_target'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['val_seq']),
        torch.FloatTensor(data['val_target'])
    )
    
    # Larger batch size for H100
    batch_size = 128 if model_type.startswith('Enhanced') else 256
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
        prefetch_factor=4  # H100 optimization
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
        prefetch_factor=4
    )
    
    # Initialize model
    input_size = data['train_seq'].shape[2]
    
    if model_type == 'LSTM':
        model = StandardLSTM(input_size=input_size, hidden_size=256, num_layers=3, dropout=0.3)
    elif model_type == 'GRU':
        model = StandardGRU(input_size=input_size, hidden_size=256, num_layers=3, dropout=0.3)
    elif model_type == 'BiLSTM':
        model = StandardBiLSTM(input_size=input_size, hidden_size=256, num_layers=3, dropout=0.3)
    elif model_type == 'BiGRU':
        model = StandardBiGRU(input_size=input_size, hidden_size=256, num_layers=3, dropout=0.3)
    elif model_type == 'Enhanced_BiLSTM':
        model = EnhancedBiLSTMAttentionComplete(
            input_size=input_size, hidden_size=256, num_layers=3, dropout=0.3, num_heads=8
        )
    elif model_type == 'Enhanced_BiGRU':
        model = EnhancedBiGRUAttentionComplete(
            input_size=input_size, hidden_size=256, num_layers=3, dropout=0.3, num_heads=8
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    print(f"Model: {model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5, fused=True)  # fused for H100
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True
    )
    
    # Training configuration
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7  # Reduced from 15
    num_epochs = 20  # Reduced from 150
    
    train_losses = []
    val_losses = []
    
    checkpoint_path = CHECKPOINT_DIR / f"{city_name}_{model_type}_checkpoint.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine if model uses physics features
    uses_physics = model_type.startswith('Enhanced')
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for sequences, targets in progress_bar:
            sequences = sequences.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if uses_physics:
                # Extract physics features from last timestep
                last_ts = sequences[:, -1, :]
                physics_features = {
                    'is_night': last_ts[:, -23:-22],
                    'hour_norm': (last_ts[:, 0:1] / 24.0),
                    'wind': last_ts[:, -15:-13],
                    'pbl_proxy': last_ts[:, -22:-21],
                    'humidity': last_ts[:, -20:-19],
                    'temperature': last_ts[:, -21:-20]
                }
                outputs, _ = model(sequences, physics_features)
            else:
                outputs, _ = model(sequences)
            
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                if uses_physics:
                    last_ts = sequences[:, -1, :]
                    physics_features = {
                        'is_night': last_ts[:, -23:-22],
                        'hour_norm': (last_ts[:, 0:1] / 24.0),
                        'wind': last_ts[:, -15:-13],
                        'pbl_proxy': last_ts[:, -22:-21],
                        'humidity': last_ts[:, -20:-19],
                        'temperature': last_ts[:, -21:-20]
                    }
                    outputs, _ = model(sequences, physics_features)
                else:
                    outputs, _ = model(sequences)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        torch.save(checkpoint, checkpoint_path)
        volume.commit()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = CHECKPOINT_DIR / f"{city_name}_{model_type}_best.pth"
            torch.save(model.state_dict(), best_model_path)
            volume.commit()
            print(f'✓ Model saved with val_loss: {val_loss:.6f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
    
    # Save results
    results = {
        'city': city_name,
        'model_type': model_type,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs_trained': len(train_losses)
    }
    
    results_file = CHECKPOINT_DIR / f"{city_name}_{model_type}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    volume.commit()
    
    print(f"\n{'='*80}")
    print(f"✓ Training complete for {city_name} - {model_type}")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"{'='*80}\n")
    
    return results

@app.local_entrypoint()
def main():
    """Main entry point - train all 8 models for Firozabad in parallel"""
    
    city = 'firozabad'
    
    # All 8 models to compare
    ml_models = ['RandomForest', 'XGBoost']
    deep_models = ['LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'Enhanced_BiLSTM', 'Enhanced_BiGRU']
    
    print("\n" + "="*80)
    print("FIROZABAD PM2.5 PREDICTION - 8 MODEL COMPARISON ON MODAL H100")
    print("="*80)
    print(f"\nML Models: {', '.join(ml_models)}")
    print(f"Deep Models: {', '.join(deep_models)}")
    print(f"Total: 8 models")
    print(f"\nTraining Strategy: PARALLEL execution for speed")
    print(f"  ML Models: Run in parallel (2 jobs)")
    print(f"  DL Models: Run in parallel (6 jobs)")
    print(f"  Epochs: 20 (reduced for faster iteration)")
    print(f"  Patience: 7 (early stopping)\n")
    
    # Train ML models in parallel
    print("\n" + "="*80)
    print("PHASE 1: TRAINING MACHINE LEARNING MODELS (PARALLEL)")
    print("="*80 + "\n")
    
    ml_jobs = []
    for model_type in ml_models:
        print(f"Starting {model_type} training job...")
        job = train_ml_model.spawn(city, model_type)
        ml_jobs.append((model_type, job))
    
    print(f"\n✓ {len(ml_jobs)} ML training jobs launched")
    print("="*80)
    print("ML TRAINING (PARALLEL EXECUTION)")
    print("="*80 + "\n")
    
    ml_results = []
    for model_type, job in ml_jobs:
        print(f"\n{'='*80}")
        print(f"WAITING FOR {model_type}")
        print(f"{'='*80}\n")
        
        # Get result (this will wait for completion)
        result = job.get()
        ml_results.append(result)
        
        print(f"\n{'='*80}")
        print(f"✓ {model_type} COMPLETED")
        print(f"  Val MSE: {result.get('val_mse', 'N/A'):.6f}")
        print(f"  Training Time: {result.get('training_time', 0):.1f}s")
        print(f"{'='*80}\n")
    
    # Train deep learning models in parallel
    print("\n" + "="*80)
    print("PHASE 2: TRAINING DEEP LEARNING MODELS (PARALLEL)")
    print("="*80 + "\n")
    
    dl_jobs = []
    for model_type in deep_models:
        print(f"Starting {model_type} training job...")
        job = train_deep_model.spawn(city, model_type)
        dl_jobs.append((model_type, job))
    
    print(f"\n✓ {len(dl_jobs)} DL training jobs launched")
    print("="*80)
    print("DEEP LEARNING TRAINING (PARALLEL EXECUTION)")
    print("="*80 + "\n")
    
    dl_results = []
    for model_type, job in dl_jobs:
        print(f"\n{'='*80}")
        print(f"WAITING FOR {model_type}")
        print(f"{'='*80}\n")
        
        # Get result (this will wait for completion)
        result = job.get()
        dl_results.append(result)
        
        print(f"\n{'='*80}")
        print(f"✓ {model_type} COMPLETED")
        print(f"  Best Val Loss: {result.get('best_val_loss', 'N/A'):.6f}")
        print(f"  Epochs Trained: {result.get('epochs_trained', 0)}")
        print(f"{'='*80}\n")
    
    all_results = ml_results + dl_results
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETED - RESULTS SUMMARY")
    print("="*80 + "\n")
    
    # Print summary
    print(f"{'Model':<25} {'Val Loss':<15} {'Training Time':<15} {'Status':<10}")
    print("-"*70)
    for result in all_results:
        if 'val_mse' in result:
            time_str = f"{result.get('training_time', 0):.1f}s"
            print(f"{result['model_type']:<25} {result['val_mse']:<15.6f} {time_str:<15} {'✓ Complete':<10}")
        else:
            epochs = result.get('epochs_trained', 0)
            time_str = f"{epochs} epochs"
            print(f"{result['model_type']:<25} {result['best_val_loss']:<15.6f} {time_str:<15} {'✓ Complete':<10}")
    
    print("\n" + "="*80)
    print("✓ ALL 8 MODELS TRAINED SUCCESSFULLY")
    print("="*80)
    print(f"\nModels saved in Modal volume: ai_ml_firozabad/checkpoints_firozabad/")
    print(f"Next step: Run inference with 'modal run modal_inferencing.py'")
    print("="*80 + "\n")

