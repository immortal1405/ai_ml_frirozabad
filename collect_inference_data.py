"""
Collect Fresh Data for Inference Period (Nov 9-11, 2025)
This script fetches the latest data for the test period
"""
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

CITY = 'firozabad'
LAT = 27.1591
LON = 78.3957

# Inference period - need 7 days before Nov 9 for 168-hour lookback
START_DATE = '2025-11-02'  # 7 days before Nov 9
END_DATE = '2025-11-11'     # End of test period

# Must match training data parameters
AIR_QUALITY_PARAMETERS = [
    'pm2_5',
    'pm10',
    'carbon_monoxide',
    'nitrogen_dioxide',
    'sulphur_dioxide',
    'ozone',
    'dust',
    'aerosol_optical_depth',
    'uv_index',
    'uv_index_clear_sky'
]

WEATHER_PARAMETERS = [
    'temperature_2m',
    'relative_humidity_2m',
    'apparent_temperature',
    'precipitation',
    'wind_speed_10m',
    'wind_direction_10m',
    'wind_gusts_10m',
    'pressure_msl',
    'surface_pressure',
    'cloud_cover',
    'cloud_cover_low',
    'cloud_cover_mid',
    'cloud_cover_high',
    'et0_fao_evapotranspiration',
    'vapor_pressure_deficit',
    'shortwave_radiation',
    'direct_radiation',
    'diffuse_radiation',
    'direct_normal_irradiance',
    'global_tilted_irradiance',
    'terrestrial_radiation',
    'is_day'
]

def collect_inference_data(max_retries=3):
    """Fetch fresh data for inference period"""
    
    # Use forecast API for recent/current data
    air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    weather_url = "https://api.open-meteo.com/v1/forecast"
    
    print(f"\nFetching inference data for {CITY}...")
    print(f"Period: {START_DATE} to {END_DATE}")
    
    air_params = {
        'latitude': LAT,
        'longitude': LON,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'hourly': AIR_QUALITY_PARAMETERS,
        'timezone': 'Asia/Kolkata'
    }
    
    weather_params = {
        'latitude': LAT,
        'longitude': LON,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'hourly': WEATHER_PARAMETERS,
        'timezone': 'Asia/Kolkata'
    }
    
    for attempt in range(max_retries):
        try:
            print(f"  Air Quality: Attempt {attempt + 1}/{max_retries}...")
            air_response = requests.get(air_quality_url, params=air_params, timeout=30)
            air_response.raise_for_status()
            air_data = air_response.json()
            
            time.sleep(1)
            
            print(f"  Weather: Attempt {attempt + 1}/{max_retries}...")
            weather_response = requests.get(weather_url, params=weather_params, timeout=30)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            # Combine data
            df = pd.DataFrame()
            df['time'] = pd.to_datetime(air_data['hourly']['time'])
            
            for var in AIR_QUALITY_PARAMETERS:
                if var in air_data['hourly']:
                    df[var] = air_data['hourly'][var]
                else:
                    print(f"  ⚠ Missing: {var}")
                    df[var] = np.nan
            
            for var in WEATHER_PARAMETERS:
                if var in weather_data['hourly']:
                    df[var] = weather_data['hourly'][var]
                else:
                    print(f"  ⚠ Missing: {var}")
                    df[var] = np.nan
            
            df['city'] = CITY
            df['latitude'] = LAT
            df['longitude'] = LON
            
            print(f"  ✓ Successfully fetched {len(df)} hourly records")
            print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
            
            # Check for missing data
            missing_pct = df.isnull().sum() / len(df) * 100
            critical_missing = missing_pct[missing_pct > 50]
            if len(critical_missing) > 0:
                print(f"  ⚠ WARNING: High missing data for: {critical_missing.to_dict()}")
            
            return df
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
    
    return None

def main():
    print("="*80)
    print("INFERENCE DATA COLLECTION - NOV 2-11, 2025")
    print("="*80)
    print(f"\nCity: {CITY.upper()}")
    print(f"Coordinates: {LAT}°N, {LON}°E")
    print(f"Period: {START_DATE} to {END_DATE} (240 hours)")
    print(f"  Nov 2-8: 168 hours (7-day lookback for sequences)")
    print(f"  Nov 9-11: 72 hours (actual test period)")
    print(f"Parameters: {len(AIR_QUALITY_PARAMETERS + WEATHER_PARAMETERS)}")
    
    df = collect_inference_data()
    
    if df is not None:
        output_file = 'raw_pm25_firozabad_inference_nov9_11.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\n{'='*80}")
        print(f"✓ INFERENCE DATA COLLECTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total records: {len(df)}")
        print(f"Expected: 240 hours (Nov 2-11, 2025)")
        print(f"  Lookback data: 168 hours (Nov 2-8)")
        print(f"  Test data: 72 hours (Nov 9-11)")
        print(f"Actual: {len(df)} hours")
        
        if len(df) < 240:
            print(f"⚠ WARNING: Expected 240 records, got {len(df)}")
        
        feature_cols = [c for c in df.columns 
                       if c not in ['time', 'city', 'latitude', 'longitude']]
        print(f"Features collected: {len(feature_cols)}")
        
        # Show PM2.5 availability (our target)
        pm25_missing = df['pm2_5'].isnull().sum()
        print(f"\nPM2.5 data:")
        print(f"  Available: {len(df) - pm25_missing} / {len(df)} hours")
        print(f"  Missing: {pm25_missing} hours ({pm25_missing/len(df)*100:.1f}%)")
        
        if pm25_missing > 0:
            print(f"  ⚠ NOTE: Missing PM2.5 values will be interpolated during preprocessing")
        
        print(f"\n✓ File saved: {output_file}")
        print(f"\nNext steps:")
        print(f"  1. Run preprocessing on this file")
        print(f"  2. Upload to Modal volume")
        print(f"  3. Run inference script")
        print("="*80 + "\n")
    else:
        print("\n✗ ERROR: Failed to collect inference data")
        print("Please check your internet connection and API availability\n")

if __name__ == "__main__":
    main()
