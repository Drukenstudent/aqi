import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import lognorm

# ==========================================
# 1. LOAD & CLEAN DATA (Do this ONCE)
# ==========================================
# Use raw string (r'...') or forward slashes to avoid path errors
file_path = r'PM 2.5 Data\\hanoi-air-quality.csv' 

print(f"Loading data from: {file_path}")
df = pd.read_csv(file_path)

# FIX: Remove hidden spaces in column names (e.g., ' pm25' -> 'pm25')
df.columns = df.columns.str.strip()

# Convert columns to numeric, turning errors (like ' ') into NaN
cols_to_fix = ['pm25', 'pm10', 'o3', 'so2', 'co']
for col in cols_to_fix:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Date sorting
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

# Create Lag Variable
if 'pm25' in df.columns:
    df['PM2.5_Lag1'] = df['pm25'].shift(1)

# Drop NaNs to get the final clean dataset
df_clean = df.dropna()
print(f"Data cleaned. Rows remaining: {len(df_clean)}")

# ==========================================
# 2. PLOTTING LOOP
# ==========================================
input_vars = ['pm10', 'o3', 'so2', 'co']

# Create 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

print("Starting Plot Loop...")

# Use 'var_name' to avoid confusing it with previous variables
for i, var_name in enumerate(input_vars):
    ax = axes[i]
    
    # Check if the column exists
    if var_name not in df_clean.columns:
        print(f"Skipping {var_name}: Not found in CSV columns: {df_clean.columns.tolist()}")
        ax.text(0.5, 0.5, f"{var_name} Not Found", ha='center')
        continue

    # Get Data (Replace 0 with epsilon for Log-Normal fitting)
    data = df_clean[var_name].replace(0, 0.001)
    
    if len(data) == 0:
        print(f"WARNING: No data for {var_name}")
        continue

    print(f"Plotting {var_name}...")

    # A. Plot Histogram (The Bars)
    ax.hist(data, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Historical Data')
    
    # B. Fit Log-Normal Curve (The Red Line)
    # floc=0 forces the curve to start at 0 (Physical reality)
    s, loc, scale = lognorm.fit(data, floc=0)
    
    # C. Generate the Line
    x = np.linspace(data.min(), data.max(), 100)
    pdf = lognorm.pdf(x, s, loc=loc, scale=scale)
    ax.plot(x, pdf, 'r-', linewidth=2, label=f'Log-Normal (s={s:.2f})')
    
    # D. Labels
    ax.set_title(f'{var_name.upper()} Distribution', fontsize=12)
    ax.legend()

plt.tight_layout()
plt.show()