import pandas as pd
import requests
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
API_TOKEN = "c95bd6ad019cee15267a1a4dda6c9e322792598a"
CURRENT_FILE_PATH = 'PM 2.5 Data/hanoi-air-quality.csv'

LOCATIONS = {
    'PM 2.5 Data/hanoi-air-quality.csv': 'hanoi',
    'PM 2.5 Data/south,-singapore-air-quality.csv': 'singapore/south',
    'PM 2.5 Data/north,-singapore-air-quality.csv': 'singapore/north',
    'PM 2.5 Data/east,-singapore-air-quality.csv':  'singapore/east',
    'PM 2.5 Data/west,-singapore-air-quality.csv':  'singapore/west',
    'PM 2.5 Data/central,-singapore-air-quality.csv': 'singapore/central',
    'PM 2.5 Data/da-nang-air-quality.csv': 'danang',
    'PM 2.5 Data/mundka,-delhi, delhi, india-air-quality.csv': 'delhi/mundka',
}

# ==========================================
# 2. INTERNAL API UPDATER
# ==========================================
def update_all_cities():
    """Runs the update process for every city in the LOCATIONS dictionary."""
    print("Starting global update for all cities...")
    for csv_path in LOCATIONS.keys():
        _fetch_and_update(csv_path)
    print("All cities processed.\n")

def _fetch_and_update(csv_path):
    norm_path = os.path.normpath(csv_path).replace('\\', '/')
    city_name = LOCATIONS.get(norm_path) or next((v for k, v in LOCATIONS.items() if os.path.normpath(k).replace('\\', '/') == norm_path), None)
            
    if not city_name:
        return 

    print(f"🔄 Fetching all pollutants for: {city_name}...")
    try:
        url = f"https://api.waqi.info/feed/{city_name}/?token={API_TOKEN}"
        response = requests.get(url, timeout=10)
        payload = response.json()
    except Exception as e:
        print(f"❌ Connection error for {city_name}: {e}")
        return 

    if payload.get('status') != 'ok':
        print(f"⚠️ API returned error for {city_name}: {payload.get('data')}")
        return

    data_payload = payload.get('data', {})
    iaqi = data_payload.get('iaqi', {})
    time_info = data_payload.get('time', {})
    
    raw_date = time_info.get('s', '').split(' ')[0]
    if not raw_date: return
    formatted_date = raw_date.replace('-', '/') 

    # Fetching all 6 major pollutants safely using .get()
    new_row = {
        'date': formatted_date,
        'pm25': iaqi.get('pm25', {}).get('v', ''),
        'pm10': iaqi.get('pm10', {}).get('v', ''),
        'o3':   iaqi.get('o3', {}).get('v', ''),
        'no2':  iaqi.get('no2', {}).get('v', ''),
        'so2':  iaqi.get('so2', {}).get('v', ''),
        'co':   iaqi.get('co', {}).get('v', '')
    }

    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            # Check if date already exists to avoid duplicates
            existing_dates = pd.to_datetime(df['date']).dt.strftime('%Y/%m/%d')
            target_date = pd.to_datetime(formatted_date).strftime('%Y/%m/%d')
            
            if target_date in existing_dates.values:
                print(f"   └─ {formatted_date} already up-to-date.")
                return
            
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(csv_path, index=False)
            print(f"   └─ Added new data for {formatted_date}")
        else:
            print(f"   └─ CSV file not found at {csv_path}")
    except Exception as e:
        print(f"   └─ Error saving CSV: {e}")

# ==========================================
# 3. PUBLIC DATA LOADER
# ==========================================
def get_data(file_path=CURRENT_FILE_PATH, auto_update_all=True):
    # Update logic
    if auto_update_all:
        update_all_cities()
    else:
        _fetch_and_update(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Convert columns to numeric
    cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').dropna(subset=['pm25'])
    
    # Cleaning
    df = df.fillna(method='ffill').fillna(0)

    if 'pm25' in df.columns:
        df['PM2.5_Lag1'] = df['pm25'].shift(1)

    df_clean = df.dropna()
    print(f"{file_path}: Loaded {len(df_clean)} rows.")
    return df_clean

if __name__ == "__main__":
    # When running data.py directly, it will update everything
    update_all_cities()
