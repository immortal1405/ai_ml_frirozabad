"""
Quick script to verify training and inference feature alignment
Run after both preprocessing scripts complete
"""
import pandas as pd
import sys

def verify_alignment():
    print("="*80)
    print("FEATURE ALIGNMENT VERIFICATION")
    print("="*80 + "\n")
    
    # Check if files exist
    try:
        train_df = pd.read_csv('cleaned_pm25_firozabad_data.csv')
        print(f"✓ Training data loaded: {train_df.shape}")
    except FileNotFoundError:
        print("✗ Training data not found: cleaned_pm25_firozabad_data.csv")
        print("  Run: python preprocessing.py")
        return False
    
    try:
        inference_df = pd.read_csv('cleaned_pm25_firozabad_inference_nov9_11.csv')
        print(f"✓ Inference data loaded: {inference_df.shape}")
    except FileNotFoundError:
        print("✗ Inference data not found: cleaned_pm25_firozabad_inference_nov9_11.csv")
        print("  Run: python preprocess_inference_data.py")
        return False
    
    print()
    
    # Get columns
    train_cols = set(train_df.columns)
    inference_cols = set(inference_df.columns)
    
    # Compare
    print(f"Training columns: {len(train_cols)}")
    print(f"Inference columns: {len(inference_cols)}")
    print()
    
    # Check for missing columns
    missing_in_inference = train_cols - inference_cols
    extra_in_inference = inference_cols - train_cols
    
    if missing_in_inference:
        print(f"⚠ MISSING IN INFERENCE ({len(missing_in_inference)}):")
        for col in sorted(missing_in_inference):
            print(f"  - {col}")
        print()
    
    if extra_in_inference:
        print(f"⚠ EXTRA IN INFERENCE ({len(extra_in_inference)}):")
        for col in sorted(extra_in_inference):
            print(f"  - {col}")
        print()
    
    # Overall status
    print("="*80)
    if train_cols == inference_cols:
        print("✅ PERFECT MATCH - All columns align!")
        print("="*80)
        print("\nFeature categories:")
        
        feature_cols = [c for c in train_cols if c not in ['time', 'city', 'latitude', 'longitude']]
        
        categories = {
            'Diurnal': ['hour', 'day', 'month', 'week', 'sin', 'cos', 'is_night', 'is_morning', 'is_afternoon', 'is_evening', 'diurnal_period'],
            'Physics': ['wind_u', 'wind_v', 'stability_index', 'humidity_temp', 'humidity_wind', 'pbl_proxy', 'precip_binary', 'high_radiation'],
            'Lag': ['lag_'],
            'Rolling': ['rolling_', 'diff_', 'is_rising', 'is_falling'],
            'Ratios': ['ratio'],
            'Industrial': ['industrial', 'stagnation', 'inversion', 'heat_stress', 'ventilation', 'accumulation', 'is_winter', 'is_summer', 'is_monsoon', 'is_weekend'],
            'Secondary': ['ozone_formation', 'secondary_aerosol', 'no2_o3', 'co_nox', 'total_oxidants', 'fine_particle', 'photochemistry', 'combustion'],
            'Stability': ['richardson', 'mixing_depth', 'deposition', 'turbulence', 'pressure_tendency', 'stable_atmosphere', 'washout']
        }
        
        for category, keywords in categories.items():
            count = len([c for c in feature_cols if any(kw in c for kw in keywords)])
            print(f"  {category}: {count}")
        
        print(f"\n✓ Ready for inference!")
        print(f"✓ All {len(feature_cols)} features properly aligned")
        return True
    else:
        print("❌ MISMATCH DETECTED!")
        print("="*80)
        print("\nFix required:")
        print("  1. Check preprocess_inference_data.py matches preprocessing.py")
        print("  2. Ensure all feature engineering functions are identical")
        print("  3. Re-run: python preprocess_inference_data.py")
        return False

if __name__ == "__main__":
    success = verify_alignment()
    sys.exit(0 if success else 1)
