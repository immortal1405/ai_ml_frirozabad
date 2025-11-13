"""
Step 1: Optimized Data Collection - REMOVES carbon_dioxide
"""
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

CITIES = {
    'firozabad': {'lat': 27.1591, 'lon': 78.3957}
}

START_DATE = '2022-01-01'
END_DATE = '2025-11-04'

# REMOVED: 'carbon_dioxide' (73% missing - useless)
AIR_QUALITY_PARAMETERS = [
    'pm2_5',                      # TARGET
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
    'et0_fao_evapotranspiration',  # REMOVED: 'evapotranspiration' (100% missing)
    'vapor_pressure_deficit',
    'shortwave_radiation',
    'direct_radiation',
    'diffuse_radiation',
    'direct_normal_irradiance',
    'global_tilted_irradiance',
    'terrestrial_radiation',
    'is_day'
]

def collect_open_meteo_data(city_name, lat, lon, start_date, end_date, max_retries=3):
    """Fetch HOURLY data from Open-Meteo API"""
    
    air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    
    print(f"\n  Fetching data for {city_name}...")
    
    air_params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': AIR_QUALITY_PARAMETERS,
        'timezone': 'Asia/Kolkata'
    }
    
    weather_params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': WEATHER_PARAMETERS,
        'timezone': 'Asia/Kolkata'
    }
    
    for attempt in range(max_retries):
        try:
            print(f"    Air Quality: Attempt {attempt + 1}/{max_retries}...")
            air_response = requests.get(air_quality_url, params=air_params, timeout=30)
            air_response.raise_for_status()
            air_data = air_response.json()
            
            time.sleep(1)
            
            print(f"    Weather: Attempt {attempt + 1}/{max_retries}...")
            weather_response = requests.get(weather_url, params=weather_params, timeout=30)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            # Combine data
            df = pd.DataFrame()
            df['time'] = pd.to_datetime(air_data['hourly']['time'])
            
            for var in AIR_QUALITY_PARAMETERS:
                if var in air_data['hourly']:
                    df[var] = air_data['hourly'][var]
            
            for var in WEATHER_PARAMETERS:
                if var in weather_data['hourly']:
                    df[var] = weather_data['hourly'][var]
            
            df['city'] = city_name
            df['latitude'] = lat
            df['longitude'] = lon
            
            print(f"    ✓ Successfully fetched {len(df)} hourly records")
            print(f"    ✓ {len([c for c in df.columns if c not in ['time', 'city', 'latitude', 'longitude']])} features collected")
            
            return df
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
    
    return None

def main():
    print("="*80)
    print("STEP 1: OPTIMIZED DATA COLLECTION (REMOVED carbon_dioxide)")
    print("="*80)
    print(f"\nDate Range: {START_DATE} to {END_DATE}")
    print(f"Cities: {', '.join(CITIES.keys())}")
    print(f"\nParameters (Optimized):")
    print(f"  Air Quality: {len(AIR_QUALITY_PARAMETERS)} parameters (removed carbon_dioxide)")
    print(f"  Weather: {len(WEATHER_PARAMETERS)} parameters (removed evapotranspiration)")
    print(f"  Total: {len(AIR_QUALITY_PARAMETERS) + len(WEATHER_PARAMETERS)} parameters\n")
    
    all_data = []
    
    for city, coords in CITIES.items():
        city_df = collect_open_meteo_data(
            city, coords['lat'], coords['lon'], START_DATE, END_DATE
        )
        
        if city_df is not None:
            all_data.append(city_df)
        
        time.sleep(2)
    
    if all_data:
        df_combined = pd.concat(all_data, ignore_index=True)
        df_combined = df_combined.sort_values(['city', 'time']).reset_index(drop=True)
        
        df_combined.to_csv('raw_pm25_firozabad_data.csv', index=False)
        
        print(f"\n{'='*80}")
        print(f"✓ DATA COLLECTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total records: {len(df_combined):,}")
        print(f"Date range: {df_combined['time'].min()} to {df_combined['time'].max()}")
        print(f"Cities: {sorted(df_combined['city'].unique().tolist())}")
        
        feature_cols = [c for c in df_combined.columns 
                       if c not in ['time', 'city', 'latitude', 'longitude']]
        print(f"Total features: {len(feature_cols)}")
        
        print(f"\n✓ File saved: raw_pm25_firozabad_data.csv\n")

if __name__ == "__main__":
    main()
