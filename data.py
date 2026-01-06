import pandas as pd

# 1. Load your single CSV file
# Replace 'your_file.csv' with the actual name of your file
df = pd.read_csv('PM 2.5 Data\\south,-singapore-air-quality.csv')

cols_to_convert = [" pm25", " pm10", " o3", " so2", " co"]
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# 2. Convert 'date' to datetime objects and Sort
# This ensures the days are in the correct order (Jan 1, Jan 2, etc.)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 3. Create the "Lag" (Historical PM2.5)
# Shifts the pm25 column down by 1 row so each row contains "Yesterday's PM2.5"
df['PM2.5_Lag1'] = df[' pm25'].shift(1)


# 4. Clean up
# Drop rows with missing values (like the first day, which has no "yesterday")
df_clean = df.dropna()
