import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORT YOUR DATA LOADER
try:
    import data
except ImportError:
    raise ImportError("❌ Could not import 'data.py'. Make sure this file is in the same folder!")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# We want to see if the model could have predicted 2024 using only past data.
SPLIT_YEAR = 2018
N_SIMULATIONS = 10000
np.random.seed(42) # Freeze randomness so results are consistent

# ==============================================================================
# 1. LOAD & SPLIT DATA
# ==============================================================================
print("--- [Phase 1] Loading & Splitting Data ---")
try:
    df_all = data.get_data()
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Ensure we have dates to split by
if 'date' not in df_all.columns:
    print("❌ Error: Your dataset is missing the 'date' column.")
    exit()

df_all['date'] = pd.to_datetime(df_all['date'])

# SPLIT THE DATA
# Train = History (2014 - 2026)
# Test  = Reality (SPLITYEAR)
train_df = df_all[df_all['date'].dt.year < SPLIT_YEAR].copy()
test_df  = df_all[df_all['date'].dt.year == SPLIT_YEAR].copy()
# 1. Sort Training Data
train_df = train_df.sort_values('date')
# 2. Moderate Time Decay (0.6 -> 1.0)
weights = np.linspace(0.6, 1.0, len(train_df))
# 3. Anomaly Penalty
train_years = train_df['date'].dt.year.values
for i, year in enumerate(train_years):
    if year in [2020, 2021]: 
        weights[i] *= 0.5
    elif year == 2015:
        weights[i] *= 0.6
print("✅ Weights applied to Training Data.")
print(f"Training Data (Pre-{SPLIT_YEAR}): {len(train_df)} days")
print(f"Testing Data (Real {SPLIT_YEAR}):   {len(test_df)} days")

if len(test_df) == 0:
    print(f"⚠️ Warning: No data found for {SPLIT_YEAR}. We cannot validate against reality.")
    # We will continue just to show the forecast, but comparison metrics will be empty.

# Define columns
target_var = 'pm25'
input_vars = ['pm10', 'o3', 'no2', 'so2', 'co', 'PM2.5_Lag1'] 
found_vars = [col for col in input_vars if col in df_all.columns]

# ==============================================================================
# 2. TRAIN THE MODEL (Using ONLY History)
# ==============================================================================
print(f"\n--- [Phase 2] Training Model (Blind to {SPLIT_YEAR}) ---")

X_train = train_df[found_vars]
Y_train = train_df[target_var]
X_const = sm.add_constant(X_train)

# Fit OLS
model = sm.WLS(Y_train, X_const, weights=weights).fit()
betas = model.params
resid_std = model.resid.std()
print(f"Model Trained. Residual Noise (Sigma): {resid_std:.2f}")

# ==============================================================================
# 3. SIMULATE 2024 (The Prediction)
# ==============================================================================
print(f"\n--- [Phase 3] Simulating {SPLIT_YEAR} Scenarios ---")

# A. Calculate Covariance from History
# Add jitter to prevent math errors (Singular Matrix)
raw_train_jitter = train_df[found_vars].replace(0, 0.001) + np.random.normal(0, 0.0001, (len(train_df), len(found_vars)))
log_data_train = np.log(np.maximum(raw_train_jitter, 0.001))

def get_weighted_stats(df, w):
    val = df.values
    avg = np.average(val, axis=0, weights=w)
    diff = val - avg
    cov = (w[:, None] * diff).T @ diff / np.sum(w)
    return avg, cov

mu_log, cov_log = get_weighted_stats(log_data_train, weights)

# FIX: Remove '.values'
cov_log[np.diag_indices_from(cov_log)] += 1e-4 

# B. Generate 10,000 Potential "2024s"
rng = np.random.default_rng(42)
# FIX: Remove '.values'
sim_log_inputs = rng.multivariate_normal(mu_log, cov_log, size=N_SIMULATIONS, method='svd')

# C. Calculate PM2.5
sim_inputs_df = pd.DataFrame(np.exp(sim_log_inputs), columns=found_vars)
sim_pm25 = np.full(N_SIMULATIONS, betas['const'])

for col in found_vars:
    sim_pm25 += betas[col] * sim_inputs_df[col].values

# D. Add Noise & Physics
noise = rng.normal(0, resid_std, N_SIMULATIONS)
sim_pm25 = np.maximum(sim_pm25 + noise, 0)

print("✅ Simulation Complete.")

# ==============================================================================
# 4. COMPARE & PLOT (The Validation)
# ==============================================================================
print(f"\n--- [Phase 4] The Verdict: Simulation vs. Reality ---")

real_values = test_df[target_var].values

# Calculate Metrics
sim_mean = np.mean(sim_pm25)
real_mean = np.mean(real_values) if len(real_values) > 0 else 0

sim_95 = np.percentile(sim_pm25, 95)
real_max = np.max(real_values) if len(real_values) > 0 else 0

sim_risk = np.mean(sim_pm25 > 55) * 100
real_risk = np.mean(real_values > 55) * 100 if len(real_values) > 0 else 0

print(f"{'METRIC':<20} | {'PREDICTED (Sim)':<15} | {'ACTUAL (2024)':<15}")
print("-" * 56)
print(f"{'Mean PM2.5':<20} | {sim_mean:<15.2f} | {real_mean:<15.2f}")
print(f"{'Risk > 55 (Unhealthy)':<20} | {sim_risk:<15.1f}% | {real_risk:<15.1f}%")
print(f"{'Extreme Cap (95%)':<20} | {sim_95:<15.2f}  | {real_max:<15.2f} (Max)")

# --- PLOT ---
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# 1. The Simulation (Forecast)
sns.histplot(sim_pm25, bins=80, stat="density", color="teal", alpha=0.3, label="Model Prediction (10k Scenarios)", kde=True)

# 2. The Reality (Actual SPLITYEAR)
if len(real_values) > 0:
    sns.kdeplot(real_values, color="black", linewidth=3, label=f"ACTUAL {SPLIT_YEAR} Data")
    # Add actual mean line
    plt.axvline(real_mean, color='black', linestyle=':', linewidth=2, label="Actual Mean")

# 3. Reference Lines
plt.axvline(sim_mean, color='teal', linestyle='--', linewidth=2, label="Predicted Mean")
plt.axvspan(55, 150, color='orange', alpha=0.1, label='Unhealthy Zone')
plt.axvspan(150, 500, color='red', alpha=0.05, label='Hazardous Zone')

# Zoom in to relevant area
max_view = max(np.percentile(sim_pm25, 99.5), real_max * 1.1)
plt.xlim(0, max_view)

plt.title(f"Validation: Did the Model Predict {SPLIT_YEAR} Correctly?", fontsize=16, weight='bold')
plt.xlabel("PM2.5 Concentration")
plt.ylabel("Probability Density")
plt.legend()
plt.tight_layout()

print("\nShowing Comparison Plot...")
plt.show()
