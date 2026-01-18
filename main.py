import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import lognorm, norm, kstest, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORT YOUR MASTER LOADER
import data 

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================
print("--- Loading Data ---")
df_clean = data.get_data()

# Define Model Variables
input_vars = ['pm10', 'o3', 'no2', 'so2', 'co', 'PM2.5_Lag1'] 
# Safety check: Only keep columns that actually exist in the CSV
input_vars = [col for col in input_vars if col in df_clean.columns]
target_var = 'pm25'

print(f"Using inputs: {input_vars}")

# ==============================================================================
# 2. DETERMINISTIC MODEL (Section 4.3 Baseline)
# ==============================================================================
print("\n--- [4.3] Training Deterministic OLS Model (Baseline) ---")
X = df_clean[input_vars]
Y = df_clean[target_var]
X = sm.add_constant(X)

# Fit OLS
model = sm.OLS(Y, X).fit()
betas = model.params

# Calculate Deterministic Error Metrics
ols_pred = model.predict(X)
rmse_ols = np.sqrt(((Y - ols_pred) ** 2).mean())
r2_ols = model.rsquared

print(model.summary())
print(f"\n[Baseline Comparison]")
print(f"Deterministic Model R²: {r2_ols:.3f}")
print(f"Deterministic Model RMSE: {rmse_ols:.2f}")

# ==============================================================================
# 3. DISTRIBUTION FITTING & JUSTIFICATION (Section 4.1)
# ==============================================================================
print("\n--- [4.1] Fitting Distributions & Generating Variates ---")
N = 10000
sim_inputs = {}
fit_metrics = []

# Set Seaborn theme
sns.set_theme(style="whitegrid", palette="muted")

# Dynamic Layout Calculation
num_vars = len(input_vars)
cols = 3
rows = (num_vars + cols - 1) // cols 

# FIGURE 1: Distribution Fits
fig_dist, axes_dist = plt.subplots(rows, cols, figsize=(15, 5 * rows), num="Figure 1: Distribution Fitting")
axes_dist = axes_dist.flatten()

for i, col in enumerate(input_vars):
    # Prepare data (replace 0 with epsilon to avoid Log-Normal errors)
    data_col = df_clean[col].replace(0, 0.001)
    
    # A. Fit Log-Normal (Proposed Model)
    shape, loc, scale = lognorm.fit(data_col, floc=0)
    d_stat_log, _ = kstest(data_col, 'lognorm', args=(shape, loc, scale))
    
    # B. Fit Normal (Standard Assumption - For Comparison)
    mu, std = norm.fit(data_col)
    d_stat_norm, _ = kstest(data_col, 'norm', args=(mu, std))
    
    # Store Metrics
    fit_metrics.append({
        'Variable': col, 
        'LogNorm D-Stat': d_stat_log, 
        'Normal D-Stat': d_stat_norm,
        'Improvement %': (d_stat_norm - d_stat_log)/d_stat_norm * 100
    })
    
    # C. Generate Random Variates for Simulation
    sim_inputs[col] = lognorm.rvs(shape, loc=loc, scale=scale, size=N)
    
    # D. Plotting
    if i < len(axes_dist):
        ax = axes_dist[i]
        sns.histplot(data_col, stat="density", color="skyblue", alpha=0.5, label="Actual Data", ax=ax)
        
        # Overlay Curves
        x_range = np.linspace(data_col.min(), data_col.max(), 100)
        ax.plot(x_range, lognorm.pdf(x_range, shape, loc, scale), 'r-', lw=2, label=f"LogNorm (D={d_stat_log:.2f})")
        ax.plot(x_range, norm.pdf(x_range, mu, std), 'g--', lw=2, label=f"Normal (D={d_stat_norm:.2f})")
        
        ax.set_title(f"{col.upper()} Distribution Fit")
        ax.legend()

# Hide empty subplots
for j in range(i + 1, len(axes_dist)):
    axes_dist[j].axis('off')

# FIX: Add vertical space (hspace) to prevent overlapping titles in the grid
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.4) 

# Print Fit Metrics Table
print("\n[Distribution Justification]")
print(pd.DataFrame(fit_metrics)[['Variable', 'LogNorm D-Stat', 'Normal D-Stat', 'Improvement %']])

# ==============================================================================
# 4. MONTE CARLO SIMULATION (Section 4.2)
# ==============================================================================
print("\n--- [4.2] Running Monte Carlo Simulation ---")

# Calculate Simulated Outcome
sim_pm25 = np.full(N, betas['const'])
for col in input_vars:
    sim_pm25 += betas[col] * sim_inputs[col]

# Enforce physical constraints
sim_pm25 = np.maximum(sim_pm25, 0)

# Calculate Statistics
mean_sim = np.mean(sim_pm25)
std_sim = np.std(sim_pm25)
lower_bound = np.percentile(sim_pm25, 2.5)
upper_bound = np.percentile(sim_pm25, 97.5)

# Risk Thresholds
risk_unhealthy = 55   
risk_extreme = 150    

prob_unhealthy = np.mean(sim_pm25 > risk_unhealthy) * 100
prob_extreme = np.mean(sim_pm25 > risk_extreme) * 100

print(f"\n[Stochastic Forecast Results]")
print(f"Mean Prediction: {mean_sim:.2f} (vs Deterministic: {ols_pred.mean():.2f})")
print(f"95% Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Probability > {risk_unhealthy} (Unhealthy): {prob_unhealthy:.2f}%")
print(f"Probability > {risk_extreme} (Extreme Event): {prob_extreme:.2f}%")

# FIGURE 2: Simulation Output
plt.figure(figsize=(12, 7), num="Figure 2: Forecast Results")

# 1. The Distribution
sns.histplot(sim_pm25, bins=60, kde=True, stat="density", 
             color="teal", alpha=0.4, linewidth=0, label="Forecast Probability")

# 2. Risk Zones
plt.axvspan(0, 12, color='green', alpha=0.05, label='Good')
plt.axvspan(12, 35, color='yellow', alpha=0.05, label='Moderate')
plt.axvspan(35, 150, color='orange', alpha=0.05, label='Unhealthy')
max_plot_val = max(np.max(sim_pm25), 200)
plt.axvspan(150, max_plot_val, color='red', alpha=0.05, label='Hazardous')

# 3. Statistical Lines
plt.axvline(mean_sim, color='blue', linestyle='-', linewidth=2, label=f'Mean Prediction ({mean_sim:.0f})')
plt.axvline(upper_bound, color='red', linestyle='--', linewidth=2, label=f'95% Worst Case ({upper_bound:.0f})')

plt.title(f'4.2 Simulation Output: Probabilistic Haze Forecast (N={N})', fontsize=16, weight='bold')
plt.xlabel('Projected PM2.5 Concentration (µg/m³)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.xlim(0, np.percentile(sim_pm25, 99.5))
plt.legend(loc='upper right', frameon=True)

# FIX: Add top spacing for title
plt.tight_layout()
plt.subplots_adjust(top=0.93)

# ==============================================================================
# 5. SENSITIVITY ANALYSIS (Section 4.4)
# ==============================================================================
print("\n--- [4.4] Sensitivity Analysis (Rank Correlation) ---")

correlations = {}
for col in input_vars:
    corr, _ = spearmanr(sim_inputs[col], sim_pm25)
    correlations[col] = corr

sensitivity_df = pd.DataFrame(list(correlations.items()), columns=['Input', 'Correlation'])
sensitivity_df['Abs_Impact'] = sensitivity_df['Correlation'].abs()
sensitivity_df = sensitivity_df.sort_values('Abs_Impact', ascending=False)

print("\n[Sensitivity Drivers]")
print(sensitivity_df[['Input', 'Correlation']])

# FIGURE 3: Tornado Chart
plt.figure(figsize=(10, 6), num="Figure 3: Sensitivity Analysis")
sns.barplot(x='Correlation', y='Input', data=sensitivity_df, palette='coolwarm')
plt.title("4.4 Sensitivity Analysis: Which variables drive PM2.5 Risk?", fontsize=14, weight='bold')
plt.xlabel("Spearman Rank Correlation with PM2.5 Output")
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# FIX: Add top spacing for title
plt.tight_layout()
plt.subplots_adjust(top=0.93)

# ==============================================================================
# FINAL SHOW
# ==============================================================================
print("\nDone! Showing all analysis plots now...")
plt.show()
