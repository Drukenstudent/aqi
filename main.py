import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import lognorm, norm, kstest, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORT YOUR MASTER LOADER
try:
    import data
except ImportError:
    raise ImportError("❌ Could not import 'data.py'. Ensure both files are in the same folder.")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
FILTER_HAZE_SEASON = False 
N_SIMULATIONS = 10000

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================
print("--- Loading Data ---")
try:
    df_clean = data.get_data()
except Exception as e:
    print(f"❌ Error loading data from data.py: {e}")
    exit()

# Seasonal Filter (Optional)
if FILTER_HAZE_SEASON:
    print("⚠️ FILTER ACTIVE: Using only Haze Season (June-Sept) data.")
    if 'date' in df_clean.columns:
        df_clean['month'] = df_clean['date'].dt.month
        df_clean = df_clean[df_clean['month'].isin([6, 7, 8, 9])]
        print(f"Data filtered to {len(df_clean)} rows.")

# Define Variables
target_var = 'pm25'
input_vars = ['pm10', 'o3', 'no2', 'so2', 'co', 'PM2.5_Lag1'] 

# Safety Check
found_vars = [col for col in input_vars if col in df_clean.columns]
if not found_vars:
    print("❌ Critical Error: No input variables found in dataset.")
    exit()

print(f"Using inputs: {found_vars}")

# ==============================================================================
# 2. DETERMINISTIC MODEL (OLS Baseline)
# ==============================================================================
print("\n--- [Phase 1] Training Deterministic Model ---")
X = df_clean[found_vars]
Y = df_clean[target_var]
X_const = sm.add_constant(X)

try:
    model = sm.OLS(Y, X_const).fit()
    betas = model.params
    resid_std = model.resid.std()
    print(f"Model Residual Std Dev (Noise): {resid_std:.2f}")
except Exception as e:
    print(f"❌ Error training OLS model: {e}")
    exit()

# ==============================================================================
# 3. DISTRIBUTION FITTING (Figure 1)
# ==============================================================================
print("\n--- [Phase 2] Fitting Distributions (Visual Check) ---")
sns.set_theme(style="whitegrid", palette="muted")

num_vars = len(found_vars)
cols = 3
rows = (num_vars + cols - 1) // cols 

fig_dist, axes_dist = plt.subplots(rows, cols, figsize=(15, 5 * rows), num="Figure 1: Distribution Fitting")
axes_dist = axes_dist.flatten()

fit_metrics = []

for i, col in enumerate(found_vars):
    # Handle zeros for Log-Normal
    data_col = df_clean[col].replace(0, 0.001)
    
    # Fit Distributions
    shape, loc, scale = lognorm.fit(data_col, floc=0)
    mu, std = norm.fit(data_col)
    
    # Calculate KS Error
    d_stat_log, _ = kstest(data_col, 'lognorm', args=(shape, loc, scale))
    d_stat_norm, _ = kstest(data_col, 'norm', args=(mu, std))
    
    fit_metrics.append({
        'Variable': col, 'LogNorm D-Stat': d_stat_log, 'Improvement %': (d_stat_norm - d_stat_log)/d_stat_norm * 100
    })
    
    # Plotting
    if i < len(axes_dist):
        ax = axes_dist[i]
        sns.histplot(data_col, stat="density", color="skyblue", alpha=0.5, label="Actual", ax=ax)
        x_range = np.linspace(data_col.min(), data_col.max(), 100)
        ax.plot(x_range, lognorm.pdf(x_range, shape, loc, scale), 'r-', lw=2, label="LogNorm")
        ax.plot(x_range, norm.pdf(x_range, mu, std), 'g--', lw=2, label="Normal")
        ax.set_title(f"{col.upper()} Fit")
        ax.legend()

# Clean up empty subplots
for j in range(i + 1, len(axes_dist)):
    axes_dist[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.4)

print("\n[Distribution Metrics]")
print(pd.DataFrame(fit_metrics)[['Variable', 'LogNorm D-Stat', 'Improvement %']])

# =================================================
