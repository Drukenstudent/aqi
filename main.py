
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import lognorm
import matplotlib.pyplot as plt

# IMPORT YOUR NEW MASTER LOADER
import data 

# ==============================================================================
# 1. GET DATA (One line to rule them all)
# ==============================================================================
# This triggers the API update check AND cleans the data
df_clean = data.get_data()

# ==============================================================================
# 2. TRAIN MODEL
# ==============================================================================
# Note: No spaces in column names anymore (data.py fixed them)
X = df_clean[['pm10', 'o3', 'so2', 'co', 'PM2.5_Lag1']]
Y = df_clean['pm25']

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
betas = model.params
print(model.summary())
# ==============================================================================
# SECTION 3.3: SIMULATION PROCESS
# ==============================================================================

N = 10000  # Number of iterations
sim_inputs = {} # Dictionary to store our random variates

print(f"--- Step 1 & 2: Fitting PDFs and Generating {N} Random Variates ---")

# We fit PDFs for ALL inputs, including the Lag variable
input_vars = ['pm10', 'o3', 'so2', 'co', 'PM2.5_Lag1']

for col in input_vars:
    # A. PREPARE DATA
    # Log-Normal cannot handle 0. We replace 0 with a tiny epsilon (0.001) for fitting.
    data_for_fitting = df_clean[col].replace(0, 0.001)
    
    # B. STEP 1: FIT HISTORICAL DATA TO PDF
    # Crucial Fix: floc=0 forces the distribution to start at 0 (Physical Reality)
    s, loc, scale = lognorm.fit(data_for_fitting, floc=0)
    
    # C. STEP 2: GENERATE RANDOM VARIATES
    # Generate N random numbers based on the fitted curve
    random_variates = lognorm.rvs(s, loc=loc, scale=scale, size=N)
    
    # Store them
    sim_inputs[col] = random_variates
    
    print(f"Fitted {col}: s={s:.3f}, scale={scale:.3f} (Loc fixed at 0)")

print("\n--- Step 3: Run the Model for N iterations ---")

# Start with the Intercept (Beta_0)
sim_pm25 = np.full(N, betas['const'])

# Add (Beta_n * Random_Variate_n) for each input
for col in input_vars:
    sim_pm25 += betas[col] * sim_inputs[col]

# OPTIONAL: Sanity check to prevent negative PM2.5 (Model artifact)
sim_pm25 = np.maximum(sim_pm25, 0)

print(f"Simulation finished. Generated {len(sim_pm25)} outcomes.")


print("\n--- Step 4: Calculate 95% Confidence Interval ---")

# Calculate Percentiles
lower_bound = np.percentile(sim_pm25, 2.5)
upper_bound = np.percentile(sim_pm25, 97.5)
mean_val = np.mean(sim_pm25)

print(f"Mean Predicted PM2.5: {mean_val:.2f}")
print(f"95% Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")

# ==============================================================================
# VISUALIZATION (For Section 4 of your Report)
# ==============================================================================

plt.figure(figsize=(10, 6))

# Histogram of results
plt.hist(sim_pm25, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Simulated PM2.5 Distribution')

# Vertical Lines for CI
plt.axvline(lower_bound, color='red', linestyle='--', linewidth=2, label='95% CI Lower')
plt.axvline(upper_bound, color='red', linestyle='--', linewidth=2, label='95% CI Upper')
plt.axvline(mean_val, color='blue', linewidth=2, label='Mean Prediction')

plt.title(f'Probabilistic Forecast of PM2.5 (N={N})', fontsize=14)
plt.xlabel('PM2.5 Concentration', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
