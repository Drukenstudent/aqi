import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import lognorm, norm, kstest, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORT YOUR MASTER LOADER
import data 

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Set to True to analyze only the Haze Season (June-Sept)
FILTER_HAZE_SEASON = False 
N_SIMULATIONS = 10000

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================
print("--- Loading Data ---")
df_clean = data.get_data()

# Optional: Seasonal Filtering
if FILTER_HAZE_SEASON:
    print("⚠️ FILTER ACTIVE: Using only Haze Season (June-Sept) data.")
    df_clean['month'] = df_clean['date'].dt.month
    df_clean = df_clean[df_clean['month'].isin([6, 7, 8, 9])]
    print(f"Data filtered to {len(df_clean)} rows.")

# Define Model Variables
input_vars = ['pm10', 'o3', 'no2', 'so2', 'co', 'PM2.5_Lag1'] 
input_vars = [col for col in input_vars if col in df_clean.columns]
target_var = 'pm25'

print(f"Using inputs: {input_vars}")

# ==============================================================================
# 2. DETERMINISTIC MODEL (OLS Baseline)
# ==============================================================================
print("\n--- [Phase 1] Training Deterministic Model ---")
X = df_clean[input_vars]
Y = df_clean[target_var]
X_const = sm.add_constant(X)

# Fit OLS
model = sm.OLS(Y, X_const).fit()
betas = model.params
resid_std = model.resid.std() # The "Unexplained Variance" (Noise)

print(model.summary())
print(f"\nModel Residual Std Dev (Noise): {resid_std:.2f}")

# ==============================================================================
# 3. DISTRIBUTION FITTING (Visual Validation - Figure 1)
# ==============================================================================
print("\n--- [Phase 2] Fitting Distributions (Visual Check) ---")
sns.set_theme(style="whitegrid", palette="muted")

# Setup Grid Layout
num_vars = len(input_vars)
cols = 3
rows = (num_vars + cols - 1) // cols 

fig_dist, axes_dist = plt.subplots(rows, cols, figsize=(15, 5 * rows), num="Figure 1: Distribution Fitting")
axes_dist = axes_dist.flatten()

fit_metrics = []

for i, col in enumerate(input_vars):
    # Replace 0 with epsilon to avoid log(0) errors
    data_col = df_clean[col].replace(0, 0.001)
    
    # Fit Log-Normal & Normal
    shape, loc, scale = lognorm.fit(data_col, floc=0)
    mu, std = norm.fit(data_col)
    
    # Calculate KS Error Statistics
    d_stat_log, _ = kstest(data_col, 'lognorm', args=(shape, loc, scale))
    d_stat_norm, _ = kstest(data_col, 'norm', args=(mu, std))
    
    fit_metrics.append({
        'Variable': col, 'LogNorm D-Stat': d_stat_log, 'Normal D-Stat': d_stat_norm,
        'Improvement %': (d_stat_norm - d_stat_log)/d_stat_norm * 100
    })
    
    # Plotting
    if i < len(axes_dist):
        ax = axes_dist[i]
        sns.histplot(data_col, stat="density", color="skyblue", alpha=0.5, label="Actual Data", ax=ax)
        
        # Overlay Curves
        x_range = np.linspace(data_col.min(), data_col.max(), 100)
        ax.plot(x_range, lognorm.pdf(x_range, shape, loc, scale), 'r-', lw=2, label=f"LogNorm (D={d_stat_log:.2f})")
        ax.plot(x_range, norm.pdf(x_range, mu, std), 'g--', lw=2, label=f"Normal (D={d_stat_norm:.2f})")
        
        ax.set_title(f"{col.upper()} Distribution Fit")
        ax.legend()

# Cleanup Empty Plots & Layout
for j in range(i + 1, len(axes_dist)):
    axes_dist[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.4) # Fix Overlapping Titles

print("\n[Distribution Metrics]")
print(pd.DataFrame(fit_metrics)[['Variable', 'LogNorm D-Stat', 'Improvement %']])

# ==============================================================================
# 4. MULTIVARIATE STOCHASTIC SIMULATION (The "Pro" Upgrade)
# ==============================================================================
print("\n--- [Phase 3] Correlated Monte Carlo Simulation ---")

# A. Prepare the Covariance Structure (Log-Space)
log_data = np.log(df_clean[input_vars].replace(0, 0.001)) 
mu_log = log_data.mean()
cov_log = log_data.cov()

# B. Generate Correlated Random Variables (The Cholesky Step)
# This replaces the independent loops.
print(f"Generating {N_SIMULATIONS} scenarios using Multivariate Log-Normal...")
sim_log_inputs = np.random.multivariate_normal(mu_log, cov_log, N_SIMULATIONS)

# C. Convert back to Real Scale
sim_inputs_df = pd.DataFrame(np.exp(sim_log_inputs), columns=input_vars)

# D. Run the Regression Equation
sim_pm25 = np.full(N_SIMULATIONS, betas['const'])

for col in input_vars:
    sim_pm25 += betas[col] * sim_inputs_df[col].values

# E. Inject Residual Noise (Model Uncertainty)
noise = np.random.normal(0, resid_std, N_SIMULATIONS)
sim_pm25 += noise

# F. Enforce Physics
sim_pm25 = np.maximum(sim_pm25, 0)

# ==============================================================================
# 5. RESULTS & VISUALIZATION
# ==============================================================================
# Statistics
mean_sim = np.mean(sim_pm25)
upper_bound = np.percentile(sim_pm25, 95)
prob_unhealthy = np.mean(sim_pm25 > 55) * 100
prob_extreme = np.mean(sim_pm25 > 150) * 100

print(f"\n[Stochastic Forecast Results]")
print(f"Mean Prediction: {mean_sim:.2f}")
print(f"95% Worst Case:  {upper_bound:.2f}")
print(f"Risk > 55 (Unhealthy): {prob_unhealthy:.2f}%")
print(f"Risk > 150 (Hazardous): {prob_extreme:.2f}%")

# FIGURE 2: Forecast Distribution
plt.figure(figsize=(12, 7), num="Figure 2: Forecast Results")
sns.histplot(sim_pm25, bins=70, kde=True, stat="density", color="teal", alpha=0.4, label="Forecast Probability")

# Risk Zones
plt.axvspan(0, 12, color='green', alpha=0.05, label='Good')
plt.axvspan(12, 35, color='yellow', alpha=0.05, label='Moderate')
plt.axvspan(35, 150, color='orange', alpha=0.05, label='Unhealthy')
# Extend Red Zone to max value
plt.axvspan(150, max(sim_pm25.max(), 200), color='red', alpha=0.05, label='Hazardous')

plt.axvline(mean_sim, color='blue', linestyle='-', linewidth=2, label=f'Mean ({mean_sim:.0f})')
plt.axvline(upper_bound, color='red', linestyle='--', linewidth=2, label=f'95% Worst Case ({upper_bound:.0f})')

plt.title("Probabilistic PM2.5 Forecast (Correlated Monte Carlo)", fontsize=16, weight='bold')
plt.xlabel("PM2.5 Concentration (µg/m³)")
plt.ylabel("Probability Density")
plt.xlim(0, np.percentile(sim_pm25, 99.5)) # Crop extreme outliers for readability
plt.legend(loc='upper right')
plt.tight_layout()
plt.subplots_adjust(top=0.93)

# FIGURE 3: Sensitivity Analysis (Tornado)
print("\n--- [Phase 4] Sensitivity Analysis ---")
correlations = {}
for col in input_vars:
    # Correlate Simulated Inputs with Simulated Output
    corr, _ = spearmanr(sim_inputs_df[col], sim_pm25)
    correlations[col] = corr

sens_df = pd.DataFrame(list(correlations.items()), columns=['Input', 'Correlation'])
sens_df['Abs_Impact'] = sens_df['Correlation'].abs()
sens_df = sens_df.sort_values('Abs_Impact', ascending=False)

plt.figure(figsize=(10, 6), num="Figure 3: Sensitivity Analysis")
sns.barplot(x='Correlation', y='Input', data=sens_df, palette='coolwarm')
plt.title("Sensitivity: Which variables drive the haze?", fontsize=14, weight='bold')
plt.xlabel("Spearman Rank Correlation with PM2.5 Output")
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.subplots_adjust(top=0.93)

print("\nDone! Showing all 3 analysis plots now...")
plt.show()
