import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

LOG_FILE = 'forecast_log.csv'

def log_forecast(mean_sim, upper_bound, prob_extreme):
    """Saves the simulated forecast to a CSV for future grading."""
    # 1. Calculate tomorrow's date
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y/%m/%d')
    
    # 2. Package the prediction
    new_log = pd.DataFrame([{
        'Date': tomorrow,
        'Predicted_Mean': round(mean_sim, 2),
        'Predicted_95th_Cap': round(upper_bound, 2),
        'Risk_Percent': round(prob_extreme, 2),
        'Actual_PM25': np.nan  # Left blank for data.py to fill tomorrow
    }])
    
    # 3. Save to log file
    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        # Prevent duplicate entries if run twice in one day
        if tomorrow not in log_df['Date'].values:
            log_df = pd.concat([log_df, new_log], ignore_index=True)
            log_df.to_csv(LOG_FILE, index=False)
            print(f"?? Telemetry: Forecast for {tomorrow} saved to log.")
    else:
        new_log.to_csv(LOG_FILE, index=False)
        print(f"?? Telemetry: Created forecast_log.csv and saved {tomorrow}.")

def log_reality(actual_pm25, formatted_date):
    """Fills in the Actual_PM25 for a given date in the log."""
    if os.path.exists(LOG_FILE) and actual_pm25 != '':
        log_df = pd.read_csv(LOG_FILE)
        
        # Find today's date where Actual_PM25 is still blank
        mask = (log_df['Date'] == formatted_date) & (log_df['Actual_PM25'].isna())
        
        if mask.any():
            log_df.loc[mask, 'Actual_PM25'] = float(actual_pm25)
            log_df.to_csv(LOG_FILE, index=False)
            print(f"?? Reality Check: Updated log! Actual PM2.5 was {actual_pm25}.")
