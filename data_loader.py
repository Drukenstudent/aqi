import pandas as pd
import requests
import sys

# --- CONFIGURATION ---
API_TOKEN = "c95bd6ad019cee15267a1a4dda6c9e322792598a"

# CSV filenames to their API endpoints
LOCATIONS = {
    # Singapore Regions
    'PM 2.5 Data/south,-singapore-air-quality.csv': 'singapore/south',
    'PM 2.5 Data/north,-singapore-air-quality.csv': 'singapore/north',
    'PM 2.5 Data/east,-singapore-air-quality.csv':  'singapore/east',
    'PM 2.5 Data/west,-singapore-air-quality.csv':  'singapore/west',
    'PM 2.5 Data/central,-singapore-air-quality.csv': 'singapore/central',

    # Vietnam
    'PM 2.5 Data/hanoi-air-quality.csv': 'hanoi',
    'PM 2.5 Data/da-nang-air-quality.csv': 'danang',
}

def fetch_and_append(csv_path, city_name):
    print(f"--- Processing {city_name} ---")
    url = f"https://api.waqi.info/feed/{city_name}/?token={API_TOKEN}"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        print(f" Network Error: {e}")
        return

    if payload.get('status') != 'ok':
        print(f" API Error: {payload.get('data')}")
        return

    # Extract Data
    data = payload.get('data', {})
    iaqi = data.get('iaqi', {})
    time_info = data.get('time', {})

    # Build the row
    new_row = {
        'date': time_info.get('s', '').split(' ')[0],  # YYYY-MM-DD
        ' pm25': iaqi.get('pm25', {}).get('v', ''),
        ' pm10': iaqi.get('pm10', {}).get('v', ''),
        ' o3':   iaqi.get('o3', {}).get('v', ''),
        ' so2':  iaqi.get('so2', {}).get('v', ''),
        ' co':   iaqi.get('co', {}).get('v', '')
    }

    if not new_row['date']:
        print("  ⚠️ Skipping: No date found in API response.")
        return

    # Update CSV
    try:
        df = pd.read_csv(csv_path)
        
        # Check if date exists (Prevent duplicates)
        if new_row['date'] in df['date'].values:
            print(f"  ℹ️  Data for {new_row['date']} already exists.")
            return

        # Append
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"  ✅ Success: Added data for {new_row['date']}")

    except FileNotFoundError:
        print(f"  ❌ Error: File {csv_path} not found.")

if __name__ == "__main__":
    print(f"Starting update for {len(LOCATIONS)} files...")
    for csv_file, city in LOCATIONS.items():
        fetch_and_append(csv_file, city)
