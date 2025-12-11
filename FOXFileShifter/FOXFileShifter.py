import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_and_process_csv(filepath, target_timestamps, param_name):
    """
    Loads CSV, fixes year, converts units, adjusts timezone, and interpolates to target timestamps.
    """
    print(f"Processing {param_name} from {filepath}...")
    
    # Load CSV
    # FIX 1: Add index_col=False. 
    try:
        df = pd.read_csv(filepath, sep=',', quotechar='"', index_col=False)
        if len(df.columns) < 2: 
            raise ValueError("Comma sep failed")
    except Exception:
        # Fallback for semicolon
        df = pd.read_csv(filepath, sep=';', quotechar='"', index_col=False)

    # Clean column names
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Parse timestamps
    df['Zeitstempel'] = pd.to_datetime(df['Zeitstempel'])
    
    # --- Timezone Adjustment (UTC to UTC+1) ---
    # DWD data is in UTC. The simulation file is in UTC+1.
    print(f"  Shifting {param_name} timestamps by +1 hour (UTC -> UTC+1)...")
    df['Zeitstempel'] = df['Zeitstempel'] + timedelta(hours=1)
    
    # Ensure values are numeric
    df['Wert'] = pd.to_numeric(df['Wert'], errors='coerce').fillna(0)

    # --- Year Alignment ---
    target_year = target_timestamps[0].year
    
    if not df.empty:
        csv_year = df['Zeitstempel'].dt.year.mode()[0]
        
        # sanity check
        if csv_year == 1970 and target_year != 1970:
            print("  Warning: CSV year detected as 1970. Check CSV formatting.")

        year_offset = target_year - csv_year
        if year_offset != 0:
            print(f"  Adjusting {param_name} CSV year by {year_offset} years to match JSON...")
            df['Zeitstempel'] = df['Zeitstempel'].apply(lambda dt: dt.replace(year=dt.year + year_offset))

    # Set index
    df = df.set_index('Zeitstempel').sort_index()

    # --- FIX 2: Remove Duplicates ---
    if df.index.duplicated().any():
        print(f"  Warning: Duplicate timestamps found in {filepath}. Removing duplicates.")
        df = df[~df.index.duplicated(keep='first')]

    # --- Unit Conversion ---
    # Input: J/cm² per 10 minutes -> Output: W/m²
    df['Watts'] = df['Wert'] * 10000 / 600

    # --- Interpolation ---
    combined_index = df.index.union(target_timestamps).sort_values()
    
    if combined_index.duplicated().any():
        combined_index = combined_index.unique()

    df_reindexed = df.reindex(combined_index)
    df_interpolated = df_reindexed['Watts'].interpolate(method='time')
    
    valid_targets = [t for t in target_timestamps if t in df_interpolated.index]
    result = df_interpolated.loc[valid_targets]
    
    return result

def shift_weather_data(data, csv_rad_g, csv_rad_l):
    """
    Original logic: Shifts internal data by 2 hours and fills gaps with CSV data.
    """
    timestep_list = data['timestepList']
    keys_to_shift = ["swDir", "swDif", "lwRad"]
    
    fmt = "%Y-%m-%d %H:%M:%S"
    def get_dt(step):
        return datetime.strptime(f"{step['date']} {step['time']}", fmt)

    t0 = get_dt(timestep_list[0])
    t1 = get_dt(timestep_list[1])
    interval_minutes = (t1 - t0).seconds / 60
    
    shift_hours = 2
    shift_steps = int((shift_hours * 60) / interval_minutes)
    
    print(f"Detected interval: {interval_minutes} minutes.")
    print(f"Shifting data by {shift_hours} hours ({shift_steps} timesteps).")

    original_values = []
    for step in timestep_list:
        original_values.append({k: step[k] for k in keys_to_shift})

    total_steps = len(timestep_list)
    
    # Identify Gap Timestamps
    gap_indices = range(total_steps - shift_steps, total_steps)
    gap_timestamps = [get_dt(timestep_list[i]) for i in gap_indices if i >= 0]

    # Process CSVs for Gaps
    fill_data = {}
    if gap_timestamps and os.path.exists(csv_rad_g) and os.path.exists(csv_rad_l):
        print(f"Filling {len(gap_timestamps)} gap steps using CSV data...")
        s_rad_g = load_and_process_csv(csv_rad_g, gap_timestamps, "RAD-G (Global)")
        s_rad_l = load_and_process_csv(csv_rad_l, gap_timestamps, "RAD-L (Longwave)")
        
        for dt in gap_timestamps:
            g_val = s_rad_g.loc[dt] if dt in s_rad_g else 0
            if pd.isna(g_val): g_val = 0
            l_val = s_rad_l.loc[dt] if dt in s_rad_l else 0
            if pd.isna(l_val): l_val = 0
            
            fill_data[dt] = {"swDir": g_val * 0.75, "swDif": g_val * 0.25, "lwRad": l_val}
    else:
        print("Warning: CSV files not found or no gaps to fill. Filling gaps with 0.")

    # Apply Changes
    for i in range(total_steps):
        source_index = i + shift_steps
        if source_index < total_steps:
            for key in keys_to_shift:
                timestep_list[i][key] = original_values[source_index][key]
        else:
            current_dt = get_dt(timestep_list[i])
            if current_dt in fill_data:
                vals = fill_data[current_dt]
                timestep_list[i]["swDir"] = vals["swDir"]
                timestep_list[i]["swDif"] = vals["swDif"]
                timestep_list[i]["lwRad"] = vals["lwRad"]
            else:
                for key in keys_to_shift:
                    timestep_list[i][key] = 0
    return data

def exchange_weather_data(data, csv_rad_g, csv_rad_l):
    """
    New logic: Replaces ALL radiation data with values from the DWD CSVs.
    """
    timestep_list = data['timestepList']
    print(f"Exchanging ALL {len(timestep_list)} timesteps with DWD data...")

    fmt = "%Y-%m-%d %H:%M:%S"
    def get_dt(step):
        return datetime.strptime(f"{step['date']} {step['time']}", fmt)

    # Collect all timestamps from the JSON
    all_timestamps = [get_dt(step) for step in timestep_list]

    if os.path.exists(csv_rad_g) and os.path.exists(csv_rad_l):
        # Load and process data for the entire time range
        s_rad_g = load_and_process_csv(csv_rad_g, all_timestamps, "RAD-G (Global)")
        s_rad_l = load_and_process_csv(csv_rad_l, all_timestamps, "RAD-L (Longwave)")
        
        # Update the JSON structure
        count = 0
        for i, step in enumerate(timestep_list):
            dt = all_timestamps[i]
            
            # Retrieve values (default to 0 if missing)
            g_val = s_rad_g.loc[dt] if dt in s_rad_g else 0
            if pd.isna(g_val): g_val = 0
            
            l_val = s_rad_l.loc[dt] if dt in s_rad_l else 0
            if pd.isna(l_val): l_val = 0
            
            # Apply assignments
            step["swDir"] = g_val * 0.75
            step["swDif"] = g_val * 0.25
            step["lwRad"] = l_val
            count += 1
            
        print(f"Updated {count} timesteps with CSV data.")
    else:
        print("Error: One or both CSV files could not be found.")
    
    return data

def main_process(input_file, output_file, csv_g, csv_l, mode="SHIFT"):
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    if mode == "SHIFT":
        data = shift_weather_data(data, csv_g, csv_l)
    elif mode == "EXCHANGE":
        data = exchange_weather_data(data, csv_g, csv_l)
    else:
        print(f"Unknown mode: {mode}")
        return

    print(f"Saving modified data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    input_filename = r"D:\MyForcing_angepasster_Wind.json"
    output_filename = r"D:\MyForcing_shifted.json"
    
    csv_g_file = r"D:\data_OBS_DEU_PT10M_RAD-G.csv"
    csv_l_file = r"D:\data_OBS_DEU_PT10M_RAD-L.csv"

    # --- CONFIGURATION ---
    # Set this to "SHIFT" for your original shifting logic
    # Set this to "EXCHANGE" to replace all data with DWD CSV data
    MODE = "EXCHANGE" 
    
    try:
        main_process(input_filename, output_filename, csv_g_file, csv_l_file, mode=MODE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()