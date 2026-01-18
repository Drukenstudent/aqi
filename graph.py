import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import data  # Your master loader

# ==========================================
# 1. SETUP & DATA LOAD
# ==========================================
# Set the "Scientific" theme
sns.set_theme(style="whitegrid", palette="muted")

print("--- Loading Data ---")
df_clean = data.get_data()

# Filter for the variables we actually care about
potential_vars = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
plot_vars = [col for col in potential_vars if col in df_clean.columns]

print(f"Plotting variables: {plot_vars}")

# ==========================================
# 2. VISUALIZATION 1: CORRELATION HEATMAP
# ==========================================
# This helps justify the "Sensitivity Analysis" results
plt.figure(figsize=(10, 8))
corr_matrix = df_clean[plot_vars].corr()

sns.heatmap(corr_matrix, 
            annot=True,      # Show numbers
            fmt=".2f",       # 2 decimal places
            cmap="coolwarm", # Red (High Corr) to Blue (Low Corr)
            linewidths=0.5, 
            vmin=-1, vmax=1)

plt.title("Correlation Matrix: Pollutant Relationships", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# ==========================================
# 3. VISUALIZATION 2: JOINT PLOTS (Regression)
# ==========================================
# Compares your biggest driver (likely PM10) vs Target (PM2.5)
if 'pm10' in plot_vars:
    g = sns.jointplot(x="pm10", y="pm25", data=df_clean, kind="reg", 
                      height=8, color="g", scatter_kws={'alpha':0.3, 's':10})
    g.fig.suptitle("Relationship: PM10 vs PM2.5", y=1.02, fontsize=14, weight='bold')
    plt.show()

# ==========================================
# 4. VISUALIZATION 3: DISTRIBUTION MATRIX
# ==========================================
# Improved version of histograms
num_vars = len(plot_vars)
rows = (num_vars + 2) // 3
fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))
axes = axes.flatten()

for i, col in enumerate(plot_vars):
    ax = axes[i]
    
    # Histogram with Kernel Density Estimate (KDE) line
    sns.histplot(df_clean[col], kde=True, stat="density", ax=ax, color="skyblue")
    
    # Add mean line
    mean_val = df_clean[col].mean()
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
    
    ax.set_title(f"Distribution of {col.upper()}", fontsize=10)
    ax.legend()

# Hide empty subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# ==========================================
# 5. VISUALIZATION 4: TIME SERIES (Haze Episodes)
# ==========================================
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='pm25', data=df_clean, color='gray', alpha=0.6, label='Daily PM2.5')

# Highlight Extreme Events (>150)
extreme_days = df_clean[df_clean['pm25'] > 150]
plt.scatter(extreme_days['date'], extreme_days['pm25'], color='red', s=20, label='Hazardous Events (>150)')

plt.title("Historical Air Quality Timeline (Haze Identification)", fontsize=14, weight='bold')
plt.ylabel("PM2.5 Concentration")
plt.xlabel("Year")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
