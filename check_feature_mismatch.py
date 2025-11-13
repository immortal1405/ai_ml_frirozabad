"""
Diagnostic script to identify the 5 extra features in inference data
"""
import pandas as pd

print("="*80)
print("FEATURE MISMATCH DIAGNOSTIC")
print("="*80 + "\n")

# Load both datasets
try:
    train_df = pd.read_csv('cleaned_pm25_firozabad_data.csv')
    print(f"✓ Training data loaded: {train_df.shape}")
except FileNotFoundError:
    print("✗ Training data not found")
    train_df = None

try:
    inference_df = pd.read_csv('cleaned_pm25_firozabad_inference_nov9_11.csv')
    print(f"✓ Inference data loaded: {inference_df.shape}")
except FileNotFoundError:
    print("✗ Inference data not found")
    inference_df = None

if train_df is not None and inference_df is not None:
    # Get feature columns (exclude metadata and target)
    meta_cols = ['time', 'city', 'latitude', 'longitude']
    
    train_features = [c for c in train_df.columns if c not in meta_cols and c != 'pm2_5']
    inference_features = [c for c in inference_df.columns if c not in meta_cols and c != 'pm2_5']
    
    print(f"\nTraining features: {len(train_features)}")
    print(f"Inference features: {len(inference_features)}")
    print(f"Difference: {len(inference_features) - len(train_features)}\n")
    
    # Find differences
    train_set = set(train_features)
    inference_set = set(inference_features)
    
    missing_in_inference = train_set - inference_set
    extra_in_inference = inference_set - train_set
    
    if missing_in_inference:
        print(f"⚠ MISSING IN INFERENCE ({len(missing_in_inference)}):")
        for feat in sorted(missing_in_inference):
            print(f"  - {feat}")
        print()
    
    if extra_in_inference:
        print(f"⚠ EXTRA IN INFERENCE ({len(extra_in_inference)}):")
        for feat in sorted(extra_in_inference):
            print(f"  - {feat}")
        print()
    
    if not missing_in_inference and not extra_in_inference:
        print("✓ Perfect match! All features align.")
    else:
        print("="*80)
        print("SOLUTION:")
        print("="*80)
        if extra_in_inference:
            print("\nInference preprocessing is creating EXTRA features.")
            print("These features need to be REMOVED from inference preprocessing:")
            for feat in sorted(extra_in_inference):
                print(f"  - {feat}")
        
        if missing_in_inference:
            print("\nInference preprocessing is MISSING features from training.")
            print("These features need to be ADDED to inference preprocessing:")
            for feat in sorted(missing_in_inference):
                print(f"  - {feat}")
        
        print("\nThe preprocessing scripts (preprocessing.py and preprocess_inference_data.py)")
        print("must generate EXACTLY the same feature names.")
    
    print("\n" + "="*80)
else:
    print("\n✗ Cannot perform comparison - missing data files")
    print("\nRun these commands first:")
    print("  1. python data_collection.py")
    print("  2. python preprocessing.py")
    print("  3. python collect_inference_data.py")
    print("  4. python preprocess_inference_data.py")
