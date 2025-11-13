"""
Step 2: OPTIMIZED Preprocessing - FAST VERSION
- Drops carbon_dioxide (too many missing values)
- Fast vectorized missing value handling
- Takes ~2-5 minutes instead of 30+ minutes
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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

def add_diurnal_features_fast(df):
    """Add diurnal features (vectorized)"""
    
    print("\nAdding diurnal features...")
    
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
    
    # Cyclical encoding (fast numpy operations)
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
    """Add physics-informed features (vectorized)"""
    
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

def add_lag_and_rolling_fast(df):
    """Add lag and rolling features (vectorized)"""
    
    print("Adding temporal features...")
    
    # Hourly lags
    lag_windows = [1, 2, 3, 4, 6, 12, 24, 48, 72]
    
    for lag in lag_windows:
        if 'pm2_5' in df.columns:
            df[f'pm2_5_lag_{lag}h'] = df.groupby('city')['pm2_5'].shift(lag)
        if 'pm10' in df.columns:
            df[f'pm10_lag_{lag}h'] = df.groupby('city')['pm10'].shift(lag)
    
    # Rolling statistics (use pandas built-in rolling)
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

def add_industrial_features(df):
    """
    Add features specific to industrial pollution patterns (Firozabad glass industry)
    These capture conditions favorable for industrial emission accumulation
    """
    
    print("Adding industrial & pollution accumulation features...")
    
    # Industrial working hours (glass furnaces operate 24/7 but peak activity 6 AM - 10 PM)
    if 'hour' in df.columns:
        df['is_industrial_hours'] = ((df['hour'] >= 6) & (df['hour'] <= 22)).astype(int)
        df['is_peak_industrial'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    
    # Atmospheric stagnation index (low wind + high humidity = poor dispersion)
    if 'wind_speed_10m' in df.columns and 'relative_humidity_2m' in df.columns:
        # Low wind speed threshold (< 2 m/s) indicates stagnant conditions
        df['stagnation_index'] = (df['relative_humidity_2m'] / 100) * (1 / (df['wind_speed_10m'] + 0.5))
        df['is_stagnant'] = (df['wind_speed_10m'] < 2.0).astype(int)
    
    # Temperature inversion proxy (cooler surface, stable atmosphere)
    if 'temperature_2m' in df.columns and 'hour' in df.columns:
        # Night/early morning with low temps favor inversions
        df['inversion_risk'] = ((df['hour'] >= 22) | (df['hour'] <= 8)).astype(int) * \
                               (1 / (df['temperature_2m'] + 1))
    
    # Heat stress index (high temp + low wind = accumulation from furnaces)
    if 'temperature_2m' in df.columns and 'wind_speed_10m' in df.columns:
        df['heat_stress_index'] = df['temperature_2m'] / (df['wind_speed_10m'] + 1e-6)
        df['is_extreme_heat'] = (df['temperature_2m'] > 40).astype(int)
    
    # Ventilation coefficient (wind speed * mixing height proxy)
    if 'wind_speed_10m' in df.columns and 'temperature_2m' in df.columns:
        # Higher temp = higher mixing, better ventilation
        df['ventilation_coef'] = df['wind_speed_10m'] * (df['temperature_2m'] / 25.0)
        df['poor_ventilation'] = (df['ventilation_coef'] < 5).astype(int)
    
    # Pollutant accumulation potential
    if 'wind_speed_10m' in df.columns and 'precipitation' in df.columns:
        # Low wind + no rain = accumulation
        df['accumulation_potential'] = (df['wind_speed_10m'] < 3).astype(int) * \
                                       (df['precipitation'] == 0).astype(int)
    
    # Seasonal indicators (winter months have worse pollution)
    if 'month' in df.columns:
        df['is_winter'] = df['month'].isin([11, 12, 1, 2]).astype(int)
        df['is_summer'] = df['month'].isin([4, 5, 6]).astype(int)
        df['is_monsoon'] = df['month'].isin([7, 8, 9]).astype(int)
        df['is_post_monsoon'] = df['month'].isin([10, 11]).astype(int)
    
    # Weekend/weekday (different industrial activity patterns)
    if 'day_of_week' in df.columns:
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
        df['is_weekday'] = (df['day_of_week'].isin([0, 1, 2, 3, 4])).astype(int)
    
    print("  ✓ Industrial & accumulation features added")
    
    return df

def add_secondary_pollutant_features(df):
    """
    Add features for secondary pollutant formation and chemical interactions
    Important for understanding PM2.5 formation from precursors
    """
    
    print("Adding secondary pollutant & chemical reaction features...")
    
    # Ozone formation conditions (high radiation + NOx)
    if 'shortwave_radiation' in df.columns and 'nitrogen_dioxide' in df.columns:
        df['ozone_formation_potential'] = df['shortwave_radiation'] * df['nitrogen_dioxide'] / 1000
        df['high_photochemistry'] = ((df['shortwave_radiation'] > 200) & 
                                     (df['nitrogen_dioxide'] > 20)).astype(int)
    
    # Secondary aerosol formation (humidity + precursors)
    if 'relative_humidity_2m' in df.columns and 'sulphur_dioxide' in df.columns:
        df['secondary_aerosol_potential'] = (df['relative_humidity_2m'] / 100) * df['sulphur_dioxide']
    
    # NOx-O3 cycle (indicator of fresh vs aged pollution)
    if 'nitrogen_dioxide' in df.columns and 'ozone' in df.columns:
        df['no2_o3_product'] = df['nitrogen_dioxide'] * df['ozone']
        df['pollution_age_indicator'] = df['ozone'] / (df['nitrogen_dioxide'] + 1e-6)
    
    # CO/NOx ratio (combustion efficiency indicator)
    if 'carbon_monoxide' in df.columns and 'nitrogen_dioxide' in df.columns:
        df['co_nox_ratio'] = df['carbon_monoxide'] / (df['nitrogen_dioxide'] + 1e-6)
        df['incomplete_combustion'] = (df['co_nox_ratio'] > 10).astype(int)
    
    # Total oxidants (O3 + NO2 as indicator of photochemical activity)
    if 'ozone' in df.columns and 'nitrogen_dioxide' in df.columns:
        df['total_oxidants'] = df['ozone'] + df['nitrogen_dioxide']
    
    # PM2.5/PM10 ratio trends (indicates fine particle dominance)
    if 'pm2_5' in df.columns and 'pm10' in df.columns:
        # High ratio means more fine particles (secondary formation)
        df['fine_particle_dominance'] = (df['pm2_5_pm10_ratio'] > 0.6).astype(int)
    
    print("  ✓ Secondary pollutant features added")
    
    return df

def add_atmospheric_stability_features(df):
    """
    Add advanced meteorological features for atmospheric stability and dispersion
    Critical for understanding pollution accumulation vs dispersal
    """
    
    print("Adding atmospheric stability & dispersion features...")
    
    # Richardson number proxy (thermal stratification)
    if 'temperature_2m' in df.columns and 'wind_speed_10m' in df.columns:
        # Simplified bulk Richardson number
        df['richardson_proxy'] = (df['temperature_2m'] - df['temperature_2m'].rolling(24).mean()) / \
                                 (df['wind_speed_10m']**2 + 1e-6)
        df['is_stable_atmosphere'] = (df['richardson_proxy'] > 0.25).astype(int)
    
    # Wet deposition potential (rain + particle presence)
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

def main():
    print("="*80)
    print("STEP 2: FIROZABAD-OPTIMIZED PREPROCESSING")
    print("="*80)
    print("\nEnhanced for industrial city characteristics:")
    print("  • Glass manufacturing emission patterns")
    print("  • Atmospheric stagnation conditions") 
    print("  • Secondary pollutant chemistry")
    print("  • Dispersion & accumulation dynamics\n")
    
    # Load data
    df = pd.read_csv('raw_pm25_firozabad_data.csv')
    
    print(f"\nInitial data shape: {df.shape}")
    print(f"Initial PM2.5 missing: {df['pm2_5'].isnull().sum()}")
    
    # Handle missing values
    handler = FastMissingValueHandler()
    df = handler.handle_missing_values(df)
    
    # Remove outliers
    Q1 = df['pm2_5'].quantile(0.005)
    Q3 = df['pm2_5'].quantile(0.995)
    initial = len(df)
    df = df[(df['pm2_5'] >= Q1) & (df['pm2_5'] <= Q3)]
    print(f"\nOutlier removal: Dropped {initial - len(df):,} rows")
    
    # Add features
    df = add_diurnal_features_fast(df)
    df = add_physics_features_fast(df)
    df = add_lag_and_rolling_fast(df)
    df = add_industrial_features(df)
    df = add_secondary_pollutant_features(df)
    df = add_atmospheric_stability_features(df)
    
    # Final drop NaN
    initial = len(df)
    df = df.dropna()
    print(f"Final NaN cleanup: Dropped {initial - len(df):,} rows")
    
    # Save
    df.to_csv('cleaned_pm25_firozabad_data.csv', index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ PREPROCESSING COMPLETE - FIROZABAD ENHANCED")
    print(f"{'='*80}")
    print(f"Final shape: {df.shape}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nNew feature categories added:")
    print(f"  • Industrial pollution indicators (glass manufacturing specific)")
    print(f"  • Atmospheric stagnation & accumulation indices")
    print(f"  • Secondary pollutant formation features")
    print(f"  • Atmospheric stability & dispersion metrics")
    print(f"  • Enhanced temporal patterns (seasonal, weekly)")
    print(f"\nFile saved: cleaned_pm25_firozabad_data.csv\n")

if __name__ == "__main__":
    main()