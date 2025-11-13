"""
Preprocess Inference Data (Nov 2-11, 2025)
EXACT COPY of training preprocessing (preprocessing.py)
CRITICAL: Must match preprocessing.py feature engineering exactly!
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class FastMissingValueHandler:
    """
    Fast missing value handling using vectorized pandas operations
    No loops - pure vectorization for speed
    """
    
    def __init__(self, max_missing_percent=0.20):
        self.max_missing_percent = max_missing_percent
    
    def handle_missing_values(self, df):
        """
        Fast 3-step missing value handling:
        1. Time-based linear interpolation (fastest)
        2. Forward/Backward fill for remaining
        3. Drop columns with excessive missing (> 20%)
        """
        
        print("\nOptimized Missing Value Handling (Fast):")
        print("="*80)
        
        df = df.sort_values(['city', 'time']).reset_index(drop=True)
        df['time'] = pd.to_datetime(df['time'])
        
        # Drop carbon_dioxide immediately (73% missing - useless)
        if 'carbon_dioxide' in df.columns:
            print("\n✓ Dropping carbon_dioxide (73.31% missing - too high)")
            df = df.drop('carbon_dioxide', axis=1)
        
        # Drop evapotranspiration (100% missing - useless)
        if 'evapotranspiration' in df.columns:
            print("✓ Dropping evapotranspiration (100% missing - no data)")
            df = df.drop('evapotranspiration', axis=1)
        
        # Identify columns by missing percentage
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        print("\nHandling missing values by column:")
        print("-"*80)
        
        for col in numeric_cols:
            if col in ['latitude', 'longitude']:
                continue
            
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_count == 0:
                print(f"  ✓ {col:40s}: Complete (0% missing)")
                continue
            
            # Complete missing columns (< 5%)
            if missing_pct < 5:
                print(f"  ⚠ {col:40s}: {missing_pct:5.2f}% missing - Interpolating...")
                # Time-based linear interpolation (vectorized)
                df[col] = df.groupby('city')[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
                # Fill any remaining with forward/backward fill
                df[col] = df.groupby('city')[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
                print(f"      ✓ Fixed")
            
            # Partially missing columns (5-20%)
            elif missing_pct < 20:
                print(f"  ⚠ {col:40s}: {missing_pct:5.2f}% missing - Interpolating...")
                df[col] = df.groupby('city')[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
                df[col] = df.groupby('city')[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
                print(f"      ✓ Fixed")
            
            # Too many missing (> 20%)
            else:
                print(f"  ✗ {col:40s}: {missing_pct:5.2f}% missing - DROPPING")
                df = df.drop(col, axis=1)
        
        # Drop rows with NaN in target
        initial_rows = len(df)
        df = df.dropna(subset=['pm2_5'])
        dropped = initial_rows - len(df)
        
        print(f"\nDropped {dropped:,} rows with missing PM2.5")
        print(f"Final shape: {df.shape}")
        
        return df

def handle_missing_values(df):
    """Handle missing values - MUST MATCH training preprocessing exactly"""
    print("\nHandling missing values (matching training pipeline)...")
    
    df = df.sort_values('time').reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])
    
    # Drop carbon_dioxide and evapotranspiration (done in training)
    if 'carbon_dioxide' in df.columns:
        df = df.drop(columns=['carbon_dioxide'])
        print("  Dropped carbon_dioxide (matching training)")
    
    if 'evapotranspiration' in df.columns:
        df = df.drop(columns=['evapotranspiration'])
        print("  Dropped evapotranspiration (matching training)")
    
    # Identify columns by missing percentage
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col == 'pm2_5':
            continue
            
        missing_pct = df[col].isnull().sum() / len(df)
        
        if missing_pct > 0.20:
            # Drop columns with >20% missing (matching training)
            df = df.drop(columns=[col])
            print(f"  Dropped {col} ({missing_pct*100:.1f}% missing)")
        elif missing_pct > 0:
            # Interpolate + forward/backward fill (matching training)
            df[col] = df[col].interpolate(method='linear', limit=24)
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            remaining = df[col].isnull().sum()
            if remaining > 0:
                df[col] = df[col].fillna(df[col].median())
    
    # Handle PM2.5 target separately
    if 'pm2_5' in df.columns:
        pm25_missing = df['pm2_5'].isnull().sum()
        if pm25_missing > 0:
            print(f"  Interpolating pm2_5: {pm25_missing} missing values")
            df['pm2_5'] = df['pm2_5'].interpolate(method='linear', limit=6)
            df['pm2_5'] = df['pm2_5'].fillna(method='ffill').fillna(method='bfill')
    
    print(f"  ✓ Missing values handled")
    return df

def add_diurnal_features_fast(df):
    """Add diurnal features - EXACT MATCH to training"""
    print("Adding diurnal features...")
    
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    df['day_of_year'] = df['time'].dt.dayofyear
    df['week_of_year'] = df['time'].dt.isocalendar().week
    
    # Binary indicators
    df['is_night'] = ((df['hour'] >= 18) | (df['hour'] < 6)).astype(int)
    df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 10)).astype(int)
    df['is_afternoon_min'] = ((df['hour'] >= 14) & (df['hour'] <= 16)).astype(int)
    df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 20)).astype(int)
    
    # Diurnal periods
    df['diurnal_period'] = pd.cut(df['hour'], 
                                   bins=[-1, 6, 9, 12, 15, 18, 21, 24],
                                   labels=[6, 0, 1, 2, 3, 4, 5],
                                   ordered=False).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    print("  ✓ Diurnal features added")
    return df

def add_physics_features_fast(df):
    """Add physics-informed features - EXACT MATCH to training"""
    print("Adding physics-informed features...")
    
    # Wind components
    if 'wind_speed_10m' in df.columns and 'wind_direction_10m' in df.columns:
        df['wind_u'] = df['wind_speed_10m'] * np.cos(np.radians(df['wind_direction_10m']))
        df['wind_v'] = df['wind_speed_10m'] * np.sin(np.radians(df['wind_direction_10m']))
    
    # Stability Index
    if 'temperature_2m' in df.columns and 'pressure_msl' in df.columns:
        df['stability_index'] = df['temperature_2m'] / (df['pressure_msl'] + 1e-6)
    
    # Interactions
    if 'relative_humidity_2m' in df.columns and 'temperature_2m' in df.columns:
        df['humidity_temp'] = df['relative_humidity_2m'] * df['temperature_2m']
        
        if 'wind_speed_10m' in df.columns:
            df['wind_humidity'] = df['wind_speed_10m'] * df['relative_humidity_2m']
        
        df['pbl_proxy'] = df['temperature_2m'] / (df['relative_humidity_2m'] + 1e-6)
    
    # Precipitation binary
    if 'precipitation' in df.columns:
        df['precip_binary'] = (df['precipitation'] > 0).astype(int)
    
    # Radiation indicator
    if 'shortwave_radiation' in df.columns:
        df['high_radiation'] = (df['shortwave_radiation'] > 50).astype(int)
    
    print("  ✓ Physics features added")
    return df

def add_industrial_features(df):
    """Add industrial & pollution accumulation features - EXACT MATCH to training"""
    print("Adding industrial & pollution accumulation features...")
    
    # Industrial working hours
    if 'hour' in df.columns:
        df['is_industrial_hours'] = ((df['hour'] >= 6) & (df['hour'] <= 22)).astype(int)
        df['is_peak_industrial'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    
    # Atmospheric stagnation index
    if 'wind_speed_10m' in df.columns and 'relative_humidity_2m' in df.columns:
        df['stagnation_index'] = (df['relative_humidity_2m'] / 100) * (1 / (df['wind_speed_10m'] + 0.5))
        df['is_stagnant'] = (df['wind_speed_10m'] < 2.0).astype(int)
    
    # Temperature inversion proxy
    if 'temperature_2m' in df.columns and 'hour' in df.columns:
        df['inversion_risk'] = ((df['hour'] >= 22) | (df['hour'] <= 8)).astype(int) * \
                               (1 / (df['temperature_2m'] + 1))
    
    # Heat stress index
    if 'temperature_2m' in df.columns and 'wind_speed_10m' in df.columns:
        df['heat_stress_index'] = df['temperature_2m'] / (df['wind_speed_10m'] + 1e-6)
        df['is_extreme_heat'] = (df['temperature_2m'] > 40).astype(int)
    
    # Ventilation coefficient
    if 'wind_speed_10m' in df.columns and 'temperature_2m' in df.columns:
        df['ventilation_coef'] = df['wind_speed_10m'] * (df['temperature_2m'] / 25.0)
        df['poor_ventilation'] = (df['ventilation_coef'] < 5).astype(int)
    
    # Pollutant accumulation potential
    if 'wind_speed_10m' in df.columns and 'precipitation' in df.columns:
        df['accumulation_potential'] = (df['wind_speed_10m'] < 3).astype(int) * \
                                       (df['precipitation'] == 0).astype(int)
    
    # Seasonal indicators
    if 'month' in df.columns:
        df['is_winter'] = df['month'].isin([11, 12, 1, 2]).astype(int)
        df['is_summer'] = df['month'].isin([4, 5, 6]).astype(int)
        df['is_monsoon'] = df['month'].isin([7, 8, 9]).astype(int)
        df['is_post_monsoon'] = df['month'].isin([10, 11]).astype(int)
    
    # Weekend/weekday
    if 'day_of_week' in df.columns:
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
        df['is_weekday'] = (df['day_of_week'].isin([0, 1, 2, 3, 4])).astype(int)
    
    print("  ✓ Industrial & accumulation features added")
    return df

def add_secondary_pollutant_features(df):
    """Add secondary pollutant & chemical reaction features - EXACT MATCH to training"""
    print("Adding secondary pollutant & chemical reaction features...")
    
    # Ozone formation conditions
    if 'shortwave_radiation' in df.columns and 'nitrogen_dioxide' in df.columns:
        df['ozone_formation_potential'] = df['shortwave_radiation'] * df['nitrogen_dioxide'] / 1000
        df['high_photochemistry'] = ((df['shortwave_radiation'] > 200) & 
                                     (df['nitrogen_dioxide'] > 20)).astype(int)
    
    # Secondary aerosol formation
    if 'relative_humidity_2m' in df.columns and 'sulphur_dioxide' in df.columns:
        df['secondary_aerosol_potential'] = (df['relative_humidity_2m'] / 100) * df['sulphur_dioxide']
    
    # NOx-O3 cycle
    if 'nitrogen_dioxide' in df.columns and 'ozone' in df.columns:
        df['no2_o3_product'] = df['nitrogen_dioxide'] * df['ozone']
        df['pollution_age_indicator'] = df['ozone'] / (df['nitrogen_dioxide'] + 1e-6)
    
    # CO/NOx ratio
    if 'carbon_monoxide' in df.columns and 'nitrogen_dioxide' in df.columns:
        df['co_nox_ratio'] = df['carbon_monoxide'] / (df['nitrogen_dioxide'] + 1e-6)
        df['incomplete_combustion'] = (df['co_nox_ratio'] > 10).astype(int)
    
    # Total oxidants
    if 'ozone' in df.columns and 'nitrogen_dioxide' in df.columns:
        df['total_oxidants'] = df['ozone'] + df['nitrogen_dioxide']
    
    # PM2.5/PM10 ratio trends
    if 'pm2_5' in df.columns and 'pm10' in df.columns:
        df['fine_particle_dominance'] = (df['pm2_5_pm10_ratio'] > 0.6).astype(int)
    
    print("  ✓ Secondary pollutant features added")
    return df

def add_atmospheric_stability_features(df):
    """Add atmospheric stability & dispersion features - EXACT MATCH to training"""
    print("Adding atmospheric stability & dispersion features...")
    
    # Add city column for groupby operations if not exists
    if 'city' not in df.columns:
        df['city'] = 'firozabad'
    
    # Richardson number proxy
    if 'temperature_2m' in df.columns and 'wind_speed_10m' in df.columns:
        df['richardson_proxy'] = (df['temperature_2m'] - df['temperature_2m'].rolling(24).mean()) / \
                                 (df['wind_speed_10m']**2 + 1e-6)
        df['is_stable_atmosphere'] = (df['richardson_proxy'] > 0.25).astype(int)
    
    # Wet deposition potential
    if 'precipitation' in df.columns and 'pm2_5' in df.columns:
        df['wet_deposition_rate'] = df['precipitation'] * df['pm2_5'] / 100
        df['rain_washout_active'] = ((df['precipitation'] > 1) & (df['pm2_5'] > 20)).astype(int)
    
    # Dry deposition velocity proxy
    if 'wind_speed_10m' in df.columns and 'relative_humidity_2m' in df.columns:
        df['dry_deposition_proxy'] = df['wind_speed_10m'] * (df['relative_humidity_2m'] / 100)
    
    # Mixing layer depth proxy (temp + wind + radiation)
    if 'temperature_2m' in df.columns and 'wind_speed_10m' in df.columns and 'shortwave_radiation' in df.columns:
        df['mixing_depth_proxy'] = (df['temperature_2m'] * df['wind_speed_10m'] * 
                                    (df['shortwave_radiation'] / 100)) / 100
        df['shallow_mixing'] = (df['mixing_depth_proxy'] < 5).astype(int)
    
    # Turbulence indicator (wind speed variability)
    if 'wind_speed_10m' in df.columns:
        df['wind_variability'] = df.groupby('city')['wind_speed_10m'].transform(
            lambda x: x.rolling(6, min_periods=1).std()
        )
        df['high_turbulence'] = (df['wind_variability'] > 2).astype(int)
    
    # Pressure tendency (rapid changes affect dispersion)
    if 'pressure_msl' in df.columns:
        df['pressure_tendency'] = df.groupby('city')['pressure_msl'].diff(3)
        df['pressure_falling'] = (df['pressure_tendency'] < -2).astype(int)
        df['pressure_rising'] = (df['pressure_tendency'] > 2).astype(int)
    
    print("  ✓ Atmospheric stability features added")
    return df

def add_lag_and_rolling_fast(df):
    """Add lag and rolling features - EXACT MATCH to training"""
    print("Adding temporal features...")
    
    # Add city column for groupby operations (inference is single city)
    if 'city' not in df.columns:
        df['city'] = 'firozabad'
    
    # Hourly lags - NOTE: Training uses '_lag_{lag}h' format with 'h' suffix!
    lag_windows = [1, 2, 3, 4, 6, 12, 24, 48, 72]
    
    for lag in lag_windows:
        if 'pm2_5' in df.columns:
            df[f'pm2_5_lag_{lag}h'] = df.groupby('city')['pm2_5'].shift(lag)
        if 'pm10' in df.columns:
            df[f'pm10_lag_{lag}h'] = df.groupby('city')['pm10'].shift(lag)
    
    # Rolling statistics - NOTE: Training uses '_rolling_{stat}_{window}h' format with 'h' suffix!
    rolling_windows = [3, 6, 12, 24]
    
    for window in rolling_windows:
        if 'pm2_5' in df.columns:
            group = df.groupby('city')['pm2_5']
            df[f'pm2_5_rolling_mean_{window}h'] = group.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'pm2_5_rolling_std_{window}h'] = group.transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df[f'pm2_5_rolling_max_{window}h'] = group.transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
            df[f'pm2_5_rolling_min_{window}h'] = group.transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
    
    # Hourly differences
    if 'pm2_5' in df.columns:
        df['pm2_5_diff_1h'] = df.groupby('city')['pm2_5'].diff(1)
        df['pm2_5_diff_6h'] = df.groupby('city')['pm2_5'].diff(6)
        df['pm2_5_diff_24h'] = df.groupby('city')['pm2_5'].diff(24)
        
        df['is_rising'] = (df['pm2_5_diff_1h'] > 0).astype(int)
        df['is_falling'] = (df['pm2_5_diff_1h'] < 0).astype(int)
    
    # Ratios
    if 'pm2_5' in df.columns and 'pm10' in df.columns:
        df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
    
    if 'nitrogen_dioxide' in df.columns and 'ozone' in df.columns:
        df['no2_o3_ratio'] = df['nitrogen_dioxide'] / (df['ozone'] + 1e-6)
    
    print("  ✓ Temporal features added")
    return df

def main():
    print("="*80)
    print("INFERENCE DATA PREPROCESSING - NOV 2-11, 2025")
    print("="*80)
    print("\nMATCHING TRAINING PIPELINE EXACTLY:")
    print("  • Same missing value handling")
    print("  • Same feature engineering functions")
    print("  • Same feature order and names")
    print("  • Glass manufacturing emission patterns")
    print("  • Atmospheric stagnation conditions") 
    print("  • Secondary pollutant chemistry")
    print("  • Dispersion & accumulation dynamics\n")
    
    # Load data
    df = pd.read_csv('raw_pm25_firozabad_inference_nov9_11.csv')
    
    print(f"\nInitial data shape: {df.shape}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"Initial PM2.5 missing: {df['pm2_5'].isnull().sum()}")
    
    # Preprocessing pipeline - EXACT MATCH to training
    df = handle_missing_values(df)
    
    # No outlier removal for inference (only done in training)
    
    # Add features in EXACT SAME ORDER as training
    df = add_diurnal_features_fast(df)
    df = add_physics_features_fast(df)
    df = add_lag_and_rolling_fast(df)
    df = add_industrial_features(df)
    df = add_secondary_pollutant_features(df)
    df = add_atmospheric_stability_features(df)
    
    # Final NaN cleanup (fill with forward/backward fill)
    print("\nFinal NaN handling...")
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64'] and df[col].isnull().any():
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            # If still NaN (first/last rows), fill with median
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    
    print(f"  ✓ All NaN values handled")
    
    # Save processed data
    output_file = 'cleaned_pm25_firozabad_inference_nov9_11.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ PREPROCESSING COMPLETE - FIROZABAD INFERENCE")
    print(f"{'='*80}")
    print(f"Final shape: {df.shape}")
    print(f"Total features: {len(df.columns)}")
    print(f"Records: {len(df)}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Feature summary
    feature_cols = [c for c in df.columns if c not in ['time', 'city', 'latitude', 'longitude']]
    print(f"\nFeature categories added:")
    
    diurnal = len([c for c in feature_cols if any(x in c for x in ['hour', 'day', 'month', 'week', 'sin', 'cos', 'is_night', 'is_morning', 'is_afternoon', 'is_evening', 'diurnal_period'])])
    physics = len([c for c in feature_cols if any(x in c for x in ['wind_u', 'wind_v', 'stability_index', 'humidity_temp', 'humidity_wind', 'pbl_proxy', 'precip_binary', 'high_radiation'])])
    lag = len([c for c in feature_cols if 'lag_' in c])
    rolling = len([c for c in feature_cols if 'rolling_' in c or 'diff_' in c or c in ['is_rising', 'is_falling']])
    ratios = len([c for c in feature_cols if 'ratio' in c])
    industrial = len([c for c in feature_cols if any(x in c for x in ['industrial', 'stagnation', 'inversion', 'heat_stress', 'ventilation', 'accumulation', 'is_winter', 'is_summer', 'is_monsoon', 'is_weekend'])])
    secondary = len([c for c in feature_cols if any(x in c for x in ['ozone_formation', 'secondary_aerosol', 'no2_o3', 'co_nox', 'total_oxidants', 'fine_particle', 'photochemistry', 'combustion'])])
    stability = len([c for c in feature_cols if any(x in c for x in ['richardson', 'mixing_depth', 'deposition', 'turbulence', 'pressure_tendency', 'stable_atmosphere', 'washout'])])
    
    print(f"  • Diurnal & temporal: {diurnal}")
    print(f"  • Physics-informed: {physics}")
    print(f"  • Lag features: {lag}")
    print(f"  • Rolling & differences: {rolling}")
    print(f"  • Ratios: {ratios}")
    print(f"  • Industrial patterns: {industrial}")
    print(f"  • Secondary pollutants: {secondary}")
    print(f"  • Atmospheric stability: {stability}")
    print(f"  • Raw features: {len(feature_cols) - diurnal - physics - lag - rolling - ratios - industrial - secondary - stability}")
    
    print(f"\n✓ File saved: {output_file}")
    print(f"\nFeature alignment check:")
    print(f"  Training features: ~155 (check cleaned_pm25_firozabad_data.csv)")
    print(f"  Inference features: {len(df.columns)}")
    print(f"  Status: {'✓ MATCH' if len(df.columns) >= 150 else '✗ MISMATCH - check feature engineering'}")
    
    print(f"\nNext steps:")
    print(f"  1. Verify feature count matches training (~155 features)")
    print(f"  2. Upload to Modal volume: modal volume put ai_ml_firozabad {output_file} /")
    print(f"  3. Run inference: modal run modal_inferencing.py")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
