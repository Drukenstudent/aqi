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
print("Generating figures... (Windows will open shortly)")

# ==========================================
# 2. VISUALIZATION 1: CORRELATION HEATMAP
# ==========================================
plt.figure(num="Figure 1: Correlation Matrix", figsize=(10, 8))
corr_matrix = df_clean[plot_vars].corr()

sns.heatmap(corr_matrix, 
            annot=True,      # Show numbers
            fmt=".2f",       # 2 decimal places
            cmap="coolwarm", # Red (High Corr) to Blue (Low Corr)
            linewidths=0.5, 
            vmin=-1, vmax=1)

plt.title("Correlation Matrix: Pollutant Relationships", fontsize=14, weight='bold')
plt.tight_layout()

# ==========================================
# 3. VISUALIZATION 2: JOINT PLOTS (Regression)
# ==========================================
if 'pm10' in plot_vars:
    # JointGrid creates its own figure
    g = sns.jointplot(x="pm10", y="pm25", data=df_clean, kind="reg", 
                      height=8, color="g", scatter_kws={'alpha':0.3, 's':10})
    
    # --- FIX FOR OVERCROPPING ---
    # 1. Adjust the top margin to make room for the title
    g.fig.subplots_adjust(top=0.95)
    
    # 2. Place title slightly lower so it fits in the window
    g.fig.suptitle("Relationship: PM10 vs PM2.5", y=0.98, fontsize=14, weight='bold')
    
    # 3. Set Window Title
    if hasattr(g.fig.canvas.manager, 'set_window_title'):
        g.fig.canvas.manager.set_window_title("Figure 2: Joint Plot")

# ==========================================
# 4. VISUALIZATION 3: DISTRIBUTION MATRIX
# ==========================================
num_vars = len(plot_vars)
cols = 3
rows = (num_vars + cols - 1) // cols  # Math trick to always round up

fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), num="Figure 3: Distributions")
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

# Hide empty subplots if any exist
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()

# ==========================================
# 5. VISUALIZATION 4: TIME SERIES (Haze Episodes)
# ==========================================
plt.figure(num="Figure 4: Time Series", figsize=(12, 6))
sns.lineplot(x='date', y='pm25', data=df_clean, color='gray', alpha=0.6, label='Daily PM2.5')

# Highlight Extreme Events (>150)
extreme_days = df_clean[df_clean['pm25'] > 150]
plt.scatter(extreme_days['date'], extreme_days['pm25'], color='red', s=20, label='Hazardous Events (>150)')

plt.title("Historical Air Quality Timeline (Haze Identification)", fontsize=14, weight='bold')
plt.ylabel("PM2.5 Concentration")
plt.xlabel("Year")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# ==========================================
# FINAL COMMAND
# ==========================================
print("Done! Showing all plots now...")
plt.show()
