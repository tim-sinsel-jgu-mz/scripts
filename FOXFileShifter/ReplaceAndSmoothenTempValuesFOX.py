import pandas as pd
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

# File paths
csv_file = 'D:/enviprojects/Mainz_Neustadt_Validation/MedianTempDataAllStations.CSV'
json_file = 'D:/enviprojects/Mainz_Neustadt_Validation/MyForcing_shifted_newTemp.json'
output_file = 'D:/enviprojects/Mainz_Neustadt_Validation/MyForcing_shifted_newTemp_updated.json'

# 1. Load CSV Data
# ----------------
df_csv = pd.read_csv(csv_file)
df_csv['Datetime'] = pd.to_datetime(df_csv['Datetime'])
# Convert Temperature from Celsius to Kelvin
df_csv['Temp_K'] = df_csv['Temperature_measured'] + 273.15

# Get the valid time range of the CSV
csv_min_dt = df_csv['Datetime'].min()
csv_max_dt = df_csv['Datetime'].max()

# 2. Load JSON Data
# -----------------
with open(json_file, 'r') as f:
    data_json = json.load(f)

def get_json_dt(entry):
    return datetime.strptime(f"{entry['date']} {entry['time']}", "%Y-%m-%d %H:%M:%S")

# 3. Prepare Arrays
# -----------------
timesteps = data_json['timestepList']
num_steps = len(timesteps)
temps = np.zeros(num_steps)

# Extract original temperatures
for i, step in enumerate(timesteps):
    temps[i] = step['tProfile'][0]['value']

# Create masks to track inserted values
inserted_mask = np.zeros(num_steps, dtype=bool)
fixed_values = np.zeros(num_steps)

# 4. Replace Values (Correct Date Alignment)
# ------------------------------------------
# We iterate through the JSON. For each timestep, we calculate what the 
# "target" time would be in 2024 (matching the CSV's year).
# If that target time exists within the CSV's range, we replace the value.
for i, step in enumerate(timesteps):
    json_dt = get_json_dt(step)
    
    # Project the JSON date to the CSV year (2024)
    # This aligns "June 26th" in JSON to "June 26th" in CSV
    try:
        target_dt = json_dt.replace(year=2024)
    except ValueError:
        continue # Handle leap years if necessary

    # Check if this projected time falls within the CSV data range
    if csv_min_dt <= target_dt <= csv_max_dt:
        # Find the nearest timestamp in the CSV
        time_diffs = (df_csv['Datetime'] - target_dt).abs()
        nearest_idx = time_diffs.idxmin()
        
        # Get the new temperature
        val = df_csv.loc[nearest_idx, 'Temp_K']
        
        # Update our arrays
        temps[i] = val
        inserted_mask[i] = True
        fixed_values[i] = val

# 5. Smooth Surroundings
# ----------------------
# Apply Gaussian smoothing iteratively.
# In every step, we reset the inserted values to their original CSV values.
# This forces the smoothing to only affect the transitions and surrounding data.
iterations = 50
sigma = 2.0 

for _ in range(iterations):
    # Smooth the entire array
    temps = gaussian_filter1d(temps, sigma=sigma)
    # RESET the inserted region to keep it "untouched"
    temps[inserted_mask] = fixed_values[inserted_mask]

# 6. Save Output
# --------------
for i, step in enumerate(timesteps):
    step['tProfile'][0]['value'] = temps[i]

with open(output_file, 'w') as f:
    json.dump(data_json, f, indent=4)

print(f"Corrected file saved to {output_file}")