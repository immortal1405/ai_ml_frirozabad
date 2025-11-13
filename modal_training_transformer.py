"""
Step 5b: Transformer Training on Modal H100 GPU
Train standard Transformer and Enhanced Transformer (with Physics) models
Run locally with: modal run modal_training_transformer.py
"""
import modal
from pathlib import Path

# Use existing volume
volume = modal.Volume.from_name("ai_ml_firozabad", create_if_missing=True)

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "pandas==2.2.0",
        "numpy==1.26.0",
        "scikit-learn==1.4.0",
        "tqdm==4.66.0",
        "matplotlib==3.8.0",
        "joblib==1.3.2"
    )
)

app = modal.App("pm25-firozabad-transformer", image=image)

VOLUME_PATH = Path("/data")
CHECKPOINT_DIR = VOLUME_PATH / "checkpoints_firozabad"

@app.function(
    gpu="H100",
    timeout=86400,  # 24 hours
    volumes={"/data": volume},
    retries=modal.Retries(max_retries=1, initial_delay=10.0),
    cpu=16.0,
    memory=65536
)
def train_transformer(city_name, model_type='Transformer'):
    """Train Transformer models (Standard or Enhanced with Physics)"""
    
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
    from model_arch import TransformerPredictor, EnhancedTransformerPredictor
    
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
        torch.set_float32_matmul_precision('high')
    
    # Load data
    data_file = VOLUME_PATH / f"{city_name}_sequences_hourly_complete.pkl"
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data loaded:")
    print(f"  Train: {data['train_seq'].shape}")
    print(f"  Val: {data['val_seq'].shape}")
    print(f"  Test: {data['test_seq'].shape}")
    print(f"  Sequence length: {data['sequence_length']} hours\n")
    
    # DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(data['train_seq']),
        torch.FloatTensor(data['train_target'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['val_seq']),
        torch.FloatTensor(data['val_target'])
    )
    
    # Batch size for Transformer (smaller due to memory)
    batch_size = 64 if model_type.startswith('Enhanced') else 128
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
        prefetch_factor=4
    )
    
    # Initialize model
    input_size = data['train_seq'].shape[2]
    
    if model_type == 'Transformer':
        model = TransformerPredictor(
            input_size=input_size,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.3
        )
    elif model_type == 'Enhanced_Transformer':
        model = EnhancedTransformerPredictor(
            input_size=input_size,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.3
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    print(f"Model: {model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    
    # Transformer typically needs lower learning rate and warmup
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.0001,  # Lower LR for Transformer
        betas=(0.9, 0.98),  # Standard Transformer betas
        eps=1e-9,
        weight_decay=1e-5,
        fused=True
    )
    
    # Warmup scheduler
    def get_lr_multiplier(step, warmup_steps=1000, d_model=256):
        """Transformer learning rate schedule with warmup"""
        step = max(step, 1)
        return min(step ** (-0.5), step * warmup_steps ** (-1.5))
    
    # Use ReduceLROnPlateau for simplicity
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, verbose=True
    )
    
    # Training configuration
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10  # More patience for Transformer
    num_epochs = 30  # More epochs for Transformer
    
    train_losses = []
    val_losses = []
    
    checkpoint_path = CHECKPOINT_DIR / f"{city_name}_{model_type}_checkpoint.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine if model uses physics features
    uses_physics = model_type.startswith('Enhanced')
    
    print(f"Training Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: 0.0001")
    print(f"  Early Stopping Patience: {patience}")
    print(f"  Uses Physics Features: {uses_physics}\n")
    
    # Training loop
    global_step = 0
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
            
            # Gradient clipping (important for Transformer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            global_step += 1
            
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
            'val_losses': val_losses,
            'global_step': global_step
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
            print(f'  Patience: {patience_counter}/{patience}')
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
        'epochs_trained': len(train_losses),
        'total_steps': global_step
    }
    
    results_file = CHECKPOINT_DIR / f"{city_name}_{model_type}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    volume.commit()
    
    print(f"\n{'='*80}")
    print(f"✓ Training complete for {city_name} - {model_type}")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Total Steps: {global_step:,}")
    print(f"{'='*80}\n")
    
    return results

@app.local_entrypoint()
def main():
    """Main entry point - train both Transformer models"""
    
    city = 'firozabad'
    
    # Transformer models
    transformer_models = ['Transformer', 'Enhanced_Transformer']
    
    print("\n" + "="*80)
    print("FIROZABAD PM2.5 PREDICTION - TRANSFORMER MODELS ON MODAL H100")
    print("="*80)
    print(f"\nTransformer Models: {', '.join(transformer_models)}")
    print(f"Total: 2 models")
    print(f"\nTraining Strategy: SEQUENTIAL execution (Transformers are memory-intensive)")
    print(f"  Epochs: 30 (Transformers need more epochs)")
    print(f"  Patience: 10 (early stopping)")
    print(f"  Learning Rate: 0.0001 (lower than RNN models)")
    print(f"  Batch Size: 64-128 (smaller due to attention memory)\n")
    
    all_results = []
    
    # Train Transformer models sequentially
    for model_type in transformer_models:
        print("\n" + "="*80)
        print(f"TRAINING {model_type.upper()}")
        print("="*80 + "\n")
        
        result = train_transformer.remote(city, model_type)
        all_results.append(result)
        
        print(f"\n{'='*80}")
        print(f"✓ {model_type} COMPLETED")
        print(f"  Best Val Loss: {result.get('best_val_loss', 'N/A'):.6f}")
        print(f"  Epochs Trained: {result.get('epochs_trained', 0)}")
        print(f"  Total Steps: {result.get('total_steps', 0):,}")
        print(f"{'='*80}\n")
    
    print("\n" + "="*80)
    print("TRANSFORMER TRAINING COMPLETED - RESULTS SUMMARY")
    print("="*80 + "\n")
    
    # Print summary
    print(f"{'Model':<30} {'Val Loss':<15} {'Epochs':<10} {'Status':<10}")
    print("-"*70)
    for result in all_results:
        epochs = result.get('epochs_trained', 0)
        print(f"{result['model_type']:<30} {result['best_val_loss']:<15.6f} {epochs:<10} {'✓ Complete':<10}")
    
    print("\n" + "="*80)
    print("✓ ALL TRANSFORMER MODELS TRAINED SUCCESSFULLY")
    print("="*80)
    print(f"\nModels saved in Modal volume: ai_ml_firozabad/checkpoints_firozabad/")
    print(f"Files created:")
    print(f"  - firozabad_Transformer_best.pth")
    print(f"  - firozabad_Enhanced_Transformer_best.pth")
    print(f"  - firozabad_Transformer_results.json")
    print(f"  - firozabad_Enhanced_Transformer_results.json")
    print(f"\nNext step: Update inference script to include Transformer models")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
