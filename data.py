import pandas as pd
import requests
import os
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
API_TOKEN = "c95bd6ad019cee15267a1a4dda6c9e322792598a"

# Default file path (change this if you want to switch cities)
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
    
    # A. Identify City
    norm_path = os.path.normpath(csv_path).replace('\\', '/')
    city_name = None
    for key, val in LOCATIONS.items():
        if os.path.normpath(key).replace('\\', '/') == norm_path:
            city_name = val
            break
            
    if not city_name:
        # If file is not in our list, return silently
        return 

    # B. Fetch Data
    print(f"🔄 Checking updates for {city_name}...")
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
    
    # C. Format Data
    # Get raw date (e.g., "2025-01-18")
    raw_date = time_info.get('s', '').split(' ')[0]
    if not raw_date: return

    # Force slash format (e.g., "2025/01/18") to match your file
    formatted_date = raw_date.replace('-', '/') 

    new_row = {
        'date': formatted_date,
        ' pm25': iaqi.get('pm25', {}).get('v', ''),
        ' pm10': iaqi.get('pm10', {}).get('v', ''),
        ' o3':   iaqi.get('o3', {}).get('v', ''),
        ' no2':  iaqi.get('no2', {}).get('v', ''),
        ' so2':  iaqi.get('so2', {}).get('v', ''),
        ' co':   iaqi.get('co', {}).get('v', '')
    }

    # D. Save to CSV (Smart Duplicate Check)
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Check for duplicates by normalizing existing dates to slashes temporarily
            existing_dates = df['date'].astype(str).str.replace('-', '/')
            
            if formatted_date in existing_dates.values:
                print(f"✅ Data up-to-date for {formatted_date}")
                return
            
            # Append only the new, correctly formatted row
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(csv_path, index=False)
            print(f"✅ UPDATED: Added {formatted_date}")
        else:
            print(f"❌ File not found: {csv_path}")

    except Exception as e:
        print(f"❌ CSV Error: {e}")

# ==========================================
# 3. PUBLIC DATA LOADER (Called by Main)
# ==========================================
def get_data(file_path=CURRENT_FILE_PATH):
    """
    Main function to load, clean, and forward-fill data.
    """
    # 1. Update first
    _fetch_and_update(file_path)

    # 2. Load
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)

    # 3. Clean Columns
    df.columns = df.columns.str.strip() # Removes spaces from headers

    cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Handle Mixed Dates (Pandas handles both / and - automatically)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

    # 5. SMART CLEANING & FORWARD FILL
    # Step A: Drop rows only if the TARGET (pm25) is missing
    df = df.dropna(subset=['pm25'])
    
    # Step B: Forward Fill (The magic line)
    # If SO2 is missing today, use yesterday's SO2
    df = df.fillna(method='ffill')
    
    # Step C: Fill any remaining gaps (at the start of file) with 0
    df = df.fillna(0)

    # 6. Create Lag Variable (Input for tomorrow's forecast)
    if 'pm25' in df.columns:
        df['PM2.5_Lag1'] = df['pm25'].shift(1)

    # Final cleanup: The Lag shift creates one empty row at the very top.
    # We drop just that one row.
    df_clean = df.dropna()
    
    print(f"📊 Loaded {len(df_clean)} rows for model (History + New).")
    return df_clean

if __name__ == "__main__":
    # Test run
    df = get_data()
    print(df.tail())
