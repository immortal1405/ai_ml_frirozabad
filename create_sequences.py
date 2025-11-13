"""
Step 3 on Modal: Create Sequences (Manual Upload Version)
Use this if you prefer to upload cleaned data manually first
"""
import modal
from pathlib import Path
import pickle

volume = modal.Volume.from_name("ai_ml_firozabad", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.0",
        "numpy==1.26.0",
        "scikit-learn==1.4.0"
    )
)

app = modal.App("pm25-sequences-manual", image=image)

VOLUME_PATH = Path("/data")
DATA_DIR = VOLUME_PATH / ""

@app.function(
    timeout=3600,
    volumes={"/data": volume}
)
def create_sequences_on_modal():
    """Create sequences on Modal"""
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    print("\n" + "="*80)
    print("STEP 3: CREATE SEQUENCES ON MODAL - FIROZABAD")
    print("="*80 + "\n")
    
    SEQUENCE_LENGTH = 168
    PREDICTION_HORIZON = 1
    CITIES = ['firozabad']
    
    def get_feature_columns(df):
        exclude = ['time', 'city', 'latitude', 'longitude', 'hour', 'day_of_week',
                   'month', 'day_of_year', 'week_of_year', 'diurnal_period', 'hour_group']
        return [c for c in df.columns if c not in exclude]
    
    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv(DATA_DIR / 'cleaned_pm25_firozabad_data.csv')
    print(f"✓ Loaded: {df.shape}\n")
    
    # Create sequences
    for city in CITIES:
        print(f"Processing {city}...")
        
        city_df = df[df['city'] == city].sort_values('time').reset_index(drop=True)
        feature_cols = get_feature_columns(city_df)
        
        scaler_features = StandardScaler()
        scaler_target = StandardScaler()
        
        features_scaled = scaler_features.fit_transform(city_df[feature_cols])
        target_scaled = scaler_target.fit_transform(city_df[['pm2_5']])
        
        sequences = []
        targets = []
        
        for i in range(len(city_df) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1):
            seq = features_scaled[i:i+SEQUENCE_LENGTH]
            target = target_scaled[i+SEQUENCE_LENGTH+PREDICTION_HORIZON-1]
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        train_size = int(0.7 * len(sequences))
        val_size = int(0.15 * len(sequences))
        
        data = {
            'train_seq': sequences[:train_size],
            'train_target': targets[:train_size],
            'val_seq': sequences[train_size:train_size+val_size],
            'val_target': targets[train_size:train_size+val_size],
            'test_seq': sequences[train_size+val_size:],
            'test_target': targets[train_size+val_size:],
            'scaler_features': scaler_features,
            'scaler_target': scaler_target,
            'feature_cols': feature_cols,
            'time_index': city_df['time'].iloc[train_size+val_size+SEQUENCE_LENGTH:].reset_index(drop=True),
            'sequence_length': SEQUENCE_LENGTH,
            'prediction_horizon': PREDICTION_HORIZON
        }
        
        print(f"  Train: {data['train_seq'].shape}, Val: {data['val_seq'].shape}, Test: {data['test_seq'].shape}")
        
        output_file = DATA_DIR / f'{city}_sequences_hourly_complete.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"  ✓ Saved\n")
    
    print("="*80)
    print("✓ ALL SEQUENCES CREATED")
    print("="*80 + "\n")

@app.local_entrypoint()
def main():
    create_sequences_on_modal.remote()
