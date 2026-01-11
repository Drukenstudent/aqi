import pandas as pd
import update_loader # Import your settings

# ==========================================
# 1. CONFIGURATION
# ==========================================
# You only change this ONE line to switch cities now:
current_file_path = 'PM 2.5 Data\\south,-singapore-air-quality.csv'

# ==========================================
# 2. AUTO-UPDATE LOGIC (Dynamic)
# ==========================================
# We verify if this file is in your known list
if current_file_path in update_loader.LOCATIONS:
    # Automatically get the correct city name (e.g., 'singapore/south')
    city_name = update_loader.LOCATIONS[current_file_path]
    
    print(f"Detected city: {city_name}")
    print("Checking for new data...")
    
    # Run the update just for this specific file
    update_loader.fetch_and_append(current_file_path, city_name)
else:
    print(f"Warning: '{current_file_path}' not found in update_loader.LOCATIONS.")
    print("Skipping auto-update and using existing file content.")

# ==========================================
# 3. LOAD DATA (Standard Model)
# ==========================================
# Load the file we just updated
df = pd.read_csv(current_file_path)

cols_to_convert = [" pm25", " pm10", " o3", " so2", " co"]
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert 'date' to datetime objects and Sort
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Create the "Lag"
df['PM2.5_Lag1'] = df[' pm25'].shift(1)

# Clean up
df_clean = df.dropna()

print(f"Model ready. Loaded {len(df_clean)} rows for {current_file_path}.")
