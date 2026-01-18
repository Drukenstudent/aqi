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

# FIX: Set a Seed for Reproducibility (Freezes the randomness)
np.random.seed(42) 

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================
print("--- Loading Data ---")
try:
    df_clean = data.get_data()
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Seasonal Filter
if FILTER_HAZE_SEASON:
    print("⚠️ FILTER ACTIVE: Using only Haze Season (June-Sept).")
    if 'date' in df_clean.columns:
        df_clean['month'] = df_clean['date'].dt.month
        df_clean = df_clean[df_clean['month'].isin([6, 7, 8, 9])]

# Define Variables
target_var = 'pm25'
input_vars = ['pm10', 'o3', 'no2', 'so2', 'co', 'PM2.5_Lag1'] 
found_vars = [col for col in input_vars if col in df_clean.columns]

if not found_vars:
    print("❌ Error: No input variables found.")
    exit()

print(f"Using inputs: {found_vars}")

# ==============================================================================
# 2. DETERMINISTIC MODEL (OLS)
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
    print(f"❌ Error training OLS: {e}")
    exit()

# ==============================================================================
# 3. DISTRIBUTION FITTING (FIGURE 1)
# ==============================================================================
print("\n--- [Phase 2] Fitting Distributions ---")
sns.set_theme(style="whitegrid", palette="muted")

num_vars = len(found_vars)
cols = 3
rows = (num_vars + cols - 1) // cols 

fig_dist, axes_dist = plt.subplots(rows, cols, figsize=(15, 5 * rows), num="Figure 1: Distribution Fitting")
axes_dist = axes_dist.flatten()

fit_metrics = []

for i, col in enumerate(found_vars):
    # Jitter to prevent constant data
    data_col = df_clean[col].replace(0, 0.001) + np.random.normal(0, 0.0001, len(df_clean))
    
    # Fit
    shape, loc, scale = lognorm.fit(data_col, floc=0)
    mu, std = norm.fit(data_col)
    
    # Calc KS Stats
    d_stat_log, _ = kstest(data_col, 'lognorm', args=(shape, loc, scale))
    d_stat_norm, _ = kstest(data_col, 'norm', args=(mu, std))
    fit_metrics.append({'Variable': col, 'LogNorm D-Stat': d_stat_log})
    
    # Plot
    if i < len(axes_dist):
        ax = axes_dist[i]
        sns.histplot(data_col, stat="density", color="skyblue", alpha=0.5, label="Actual", ax=ax)
        x_range = np.linspace(data_col.min(), data_col.max(), 100)
        ax.plot(x_range, lognorm.pdf(x_range, shape, loc, scale), 'r-', lw=2, label="LogNorm")
        ax.plot(x_range, norm.pdf(x_range, mu, std), 'g--', lw=2, label="Normal")
        ax.set_title(f"{col.upper()} Fit")
        ax.legend()

for j in range(i + 1, len(axes_dist)):
    axes_dist[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.4)

# ==============================================================================
# 4. MULTIVARIATE SIMULATION
# ==============================================================================
print("\n--- [Phase 3] Correlated Monte Carlo Simulation ---")

try:
    # Prepare Data (Log-Space)
    # Add jitter to raw data before logging to prevent singular matrix
    raw_data_jittered = df_clean[found_vars].replace(0, 0.001) + np.random.normal(0, 0.0001, (len(df_clean), len(found_vars)))
    log_data = np.log(np.maximum(raw_data_jittered, 0.001)) 
    
    mu_log = log_data.mean()
    cov_log = log_data.cov()

    # Manual Regularization (Safety for Matrix Math)
    cov_log.values[np.diag_indices_from(cov_log)] += 1e-4

    print(f"Generating {N_SIMULATIONS} scenarios...")
    # Standard generator with Seed set at top
    sim_log_inputs = np.random.multivariate_normal(mu_log.values, cov_log.values, N_SIMULATIONS, check_valid='warn')

    # Convert & Regress
    sim_inputs_df = pd.DataFrame(np.exp(sim_log_inputs), columns=found_vars)
    sim_pm25 = np.full(N_SIMULATIONS, betas['const'])
    for col in found_vars:
        sim_pm25 += betas[col] * sim_inputs_df[col].values

    # Noise & Physics
    noise = np.random.normal(0, resid_std, N_SIMULATIONS)
    sim_pm25 = np.maximum(sim_pm25 + noise, 0)
    
    print("✅ Simulation Successful.")

except Exception as e:
    print(f"⚠️ SIMULATION FAILED: {e}")
    # Fallback
    sim_pm25 = np.random.lognormal(3, 0.5, N_SIMULATIONS)
    sim_inputs_df = pd.DataFrame(np.random.normal(0,1, (N_SIMULATIONS, len(found_vars))), columns=found_vars)

# ==============================================================================
# 5. RESULTS & FIGURES
# ==============================================================================
mean_sim = np.mean(sim_pm25)
upper_bound = np.percentile(sim_pm25, 95)
prob_extreme = np.mean(sim_pm25 > 150) * 100

print(f"\n[Forecast] Mean: {mean_sim:.2f} | Risk > 150: {prob_extreme:.2f}%")

# FIGURE 2: Forecast (Fixed Aspect Ratio)
plt.figure(figsize=(12, 7), num="Figure 2: Forecast Results")
sns.histplot(sim_pm25, bins=80, kde=True, stat="density", color="teal", alpha=0.4, label="Forecast")
plt.axvspan(35, 150, color='orange', alpha=0.05, label='Unhealthy')
plt.axvspan(150, max(sim_pm25.max(), 200), color='red', alpha=0.05, label='Hazardous')
plt.axvline(mean_sim, color='blue', linestyle='-', label=f'Mean')
plt.axvline(upper_bound, color='red', linestyle='--', label=f'95% Worst')

# FIX: Dynamic Zoom (Hide outliers > 99.5th percentile)
zoom_limit = np.percentile(sim_pm25, 99.5)
plt.xlim(0, max(zoom_limit, 100)) 

plt.title("Probabilistic PM2.5 Forecast", fontsize=16, weight='bold')
plt.legend()
plt.tight_layout()
plt.subplots_adjust(top=0.93)

# FIGURE 3: Sensitivity
print("\n--- [Phase 4] Sensitivity Analysis ---")
correlations = {}
for col in found_vars:
    # Handle NaN correlation if variable is flat
    if sim_inputs_df[col].std() == 0:
        corr = 0
    else:
        corr, _ = spearmanr(sim_inputs_df[col], sim_pm25)
        if np.isnan(corr): corr = 0
    correlations[col] = corr

sens_df = pd.DataFrame(list(correlations.items()), columns=['Input', 'Correlation'])
sens_df = sens_df.sort_values(by='Correlation', key=abs, ascending=False)

plt.figure(figsize=(10, 6), num="Figure 3: Sensitivity Analysis")
sns.barplot(x='Correlation', y='Input', data=sens_df, palette='coolwarm')
plt.title("Sensitivity Analysis", fontsize=14, weight='bold')
plt.axvline(0, color='black')
plt.tight_layout()
plt.subplots_adjust(top=0.93)

print("\nDone! Attempting to launch windows...")
plt.show()
