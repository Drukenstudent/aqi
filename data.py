import pandas as pd
import requests
import os
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
API_TOKEN = "c95bd6ad019cee15267a1a4dda6c9e322792598a"

# Default file for the project
CURRENT_FILE_PATH = 'PM 2.5 Data/hanoi-air-quality.csv'

# Map files to API Cities
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
def _fetch_and_update(csv_path):
    """Fetches data from WAQI and matches your specific CSV format."""
    
    # 1. Find the city for this file
    norm_path = os.path.normpath(csv_path).replace('\\', '/') # Unify slashes
    
    # Simple lookup - check if any key ends with the filename
    city_name = None
    for key, val in LOCATIONS.items():
        if os.path.normpath(key).replace('\\', '/') == norm_path:
            city_name = val
            break
            
    if not city_name:
        print(f"⚠️  No API mapping found for {csv_path}. Skipping update.")
        return

    # 2. Fetch Data
    print(f"🔄 Connecting to API for {city_name}...")
    try:
        url = f"https://api.waqi.info/feed/{city_name}/?token={API_TOKEN}"
        response = requests.get(url, timeout=10)
        payload = response.json()
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    if payload.get('status') != 'ok':
        return

    data = payload.get('data', {})
    iaqi = data.get('iaqi', {})
    time_info = data.get('time', {})
    
    # 3. Format Date to YYYY/MM/DD (Matches your file)
    # API gives "2025-01-11", we convert to "2025/01/11"
    raw_date = time_info.get('s', '').split(' ')[0]
    formatted_date = raw_date.replace('-', '/') 

    if not formatted_date:
        return

    # 4. Prepare Row (Included 'no2' to match your header)
    # Note: Keys match your CSV headers exactly (some might have spaces)
    new_row = {
        'date': formatted_date,
        ' pm25': iaqi.get('pm25', {}).get('v', ''),
        ' pm10': iaqi.get('pm10', {}).get('v', ''),
        ' o3':   iaqi.get('o3', {}).get('v', ''),
        ' no2':  iaqi.get('no2', {}).get('v', ''), # Added NO2
        ' so2':  iaqi.get('so2', {}).get('v', ''),
        ' co':   iaqi.get('co', {}).get('v', '')
    }

    # 5. Save to CSV
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Check for duplicates using the slash format
            # We convert column to string to ensure comparison works
            if formatted_date in df['date'].astype(str).values:
                print(f"✅ Data up-to-date for {formatted_date}.")
                return
            
            # Append
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(csv_path, index=False)
            print(f"✅ UPDATED: Added row for {formatted_date}")
        else:
            print(f"❌ File not found: {csv_path}")

    except Exception as e:
        print(f"❌ Error saving CSV: {e}")

# ==========================================
# 3. PUBLIC DATA LOADER (Called by Main)
# ==========================================
def get_data(file_path=CURRENT_FILE_PATH):
    """
    Main function to load and clean data.
    """
    # A. Update first
    _fetch_and_update(file_path)

    # B. Load File
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)

    # C. Clean Columns (Remove spaces from ' pm25', etc.)
    df.columns = df.columns.str.strip()

    # D. Convert Numerics
    # Added 'no2' to the list
    cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # E. Sort Dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

    # F. Create Lag
    if 'pm25' in df.columns:
        df['PM2.5_Lag1'] = df['pm25'].shift(1)

    df_clean = df.dropna()
    print(f"📊 Loaded {len(df_clean)} rows for model.")
    return df_clean

if __name__ == "__main__":
    df = get_data()
    print(df.tail())
