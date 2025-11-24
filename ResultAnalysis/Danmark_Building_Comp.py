import os
import re  # For parsing filenames and column headers
import glob  # For finding files
import numpy as np
import pandas as pd  # For data loading and time-series alignment
import matplotlib.pyplot as plt  # For plotting
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score  # For stats
from scipy.stats import pearsonr # For correlation r-squared

# --- Note on new libraries ---
# This script now requires pandas, matplotlib, scikit-learn, and scipy.
# You may need to install them using:
# pip install pandas matplotlib scikit-learn scipy
# -----------------------------


def extract_model_data(data_directory):
    """
    Recursively finds all *.AT_1DT files, extracts data for all
    distances (from filename) or heights (from ROOF file content),
    and returns a clean, long-format DataFrame.
    """
    print("\n--- Starting Model Data Extraction ---")
    
    # Use glob to find all .AT_1DT files recursively
    search_path = os.path.join(data_directory, '**', '*.AT_1DT')
    all_receptor_files = glob.glob(search_path, recursive=True)

    if not all_receptor_files:
        print(f"CRITICAL ERROR: No '.AT_1DT' files found in '{data_directory}'.")
        return None
        
    print(f"Found {len(all_receptor_files)} receptor files to process.")

    all_data_list = []  # List to hold all extracted data rows

    # Define the column names we need to find in the header
    z_col_name = 'z (m)'
    temp_col_name = 'Potential Air Temperature (°C)'
    dt_col_name = 'DateTime'
    
    # Define the base height of the roof to calculate distance *above* it
    roof_base_height = 21.0  # Building height in meters
    
    # Regex to find location and distance from *either* format:
    # Format 1: (NORTH)(125)(CM)
    # Format 2: (NORTH)_(05)
    re_dist = re.compile(
        r'(NORTH|SOUTH1|SOUTH2|SOUTH|ROOF)(?:(?:_(\d{2}))|(?:(\d+)CM))?', 
        re.IGNORECASE
    )

    for file_path in all_receptor_files:
        filename = os.path.basename(file_path)
        location = None
        
        # Standardize location names
        if 'NORTH' in filename.upper():
            location = 'NORTH'
        elif 'SOUTH1' in filename.upper():
            location = 'SOUTH1'
        elif 'SOUTH2' in filename.upper():
            location = 'SOUTH2'
        elif 'ROOF' in filename.upper():
            location = 'ROOF'
        elif 'SOUTH' in filename.upper(): # Fallback
            location = 'SOUTH'
        else:
            print(f"Warning: Skipping file, could not parse location: {filename}")
            continue

        print(f"Processing: {filename} (Location: {location})")

        try:
            with open(file_path, 'r', encoding='cp1252', errors='ignore') as f:
                header_line = f.readline().strip()
                header = [h.strip() for h in header_line.split(',')]

                try:
                    z_index = header.index(z_col_name)
                    temp_index = header.index(temp_col_name)
                    dt_index = header.index(dt_col_name)
                except ValueError:
                    print(f"Error: Could not find required columns in '{filename}'. Skipping.")
                    continue

                # --- Handle data extraction ---
                
                # Case 1: ROOF file (read distances from 'z (m)' column)
                if location == 'ROOF':
                    for line in f:
                        if not line.strip(): continue
                        parts = [p.strip() for p in line.strip().split(',')]
                        if len(parts) <= max(z_index, temp_index, dt_index): continue

                        try:
                            # This will correctly fail on '(in building)'
                            current_z = float(parts[z_index]) 
                            temp = float(parts[temp_index])
                            dt_val = parts[dt_index]
                            
                            # Convert absolute Z height to distance above roof
                            distance = current_z - roof_base_height
                            
                            # Skip any data from *below* the roof surface
                            if distance < -0.01: # Small tolerance
                                continue
                                
                            all_data_list.append([dt_val, location, distance, temp])
                            
                        except (ValueError, IndexError):
                            # This will correctly skip lines with non-numeric data
                            continue
                            
                # Case 2: All other files (get distance from filename)
                else:
                    # --- NEW LOGIC ---
                    match = re_dist.search(filename)
                    distance_m = None
                    if match:
                        # Check format 1: _05
                        if match.group(2):
                            # _05 -> 0.5m, _15 -> 1.5m
                            distance_m = float(match.group(2)) / 10.0
                        # Check format 2: 125CM
                        elif match.group(3):
                            # 125CM -> 1.25m, 25CM -> 0.25m
                            distance_m = float(match.group(3)) / 100.0
                    
                    if distance_m is None:
                        print(f"Warning: No distance in name for {filename}. Defaulting to distance 1.75m.")
                        distance_m = 1.75 # Default distance if not specified
                    
                    # --- FIX ---
                    # The z(m) check was WRONG. The files for facades
                    # often contain only one z-level. We will extract
                    # data from *any* z-level in these files, assuming
                    # it corresponds to the distance in the filename.
                    for line in f:
                        if not line.strip(): continue
                        parts = [p.strip() for p in line.strip().split(',')]
                        if len(parts) <= max(z_index, temp_index, dt_index): continue
                        
                        try:
                            # Just try to read the temp. If it's a number, take it.
                            temp = float(parts[temp_index])
                            dt_val = parts[dt_index]
                            
                            # Save with the distance from the FILENAME
                            all_data_list.append([dt_val, location, distance_m, temp])
                            
                            # We only need one value per timestamp
                            # (This assumes one z-level of interest per file)
                            # To be safer, we can read all and drop duplicates later
                            
                        except (ValueError, IndexError):
                            # This will skip headers or '(in building)'
                            continue
                            
        except FileNotFoundError:
            print(f"Error: File not found at '{file_path}'")
        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    if not all_data_list:
        print("CRITICAL ERROR: No model data was successfully extracted.")
        return None

    # Convert list to DataFrame
    df = pd.DataFrame(all_data_list, columns=['DateTime', 'Location', 'Distance', 'Temperature'])
    
    # Convert types
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
    df['Distance'] = pd.to_numeric(df['Distance'])
    df['Temperature'] = pd.to_numeric(df['Temperature'])
    
    # --- FIX ---
    # The facade files might have multiple z-levels (e.g., 0.25, 1.75).
    # We must drop duplicates to keep only one value per timestamp/distance pair.
    # We keep the *first* one found (usually the lowest z-level).
    df.drop_duplicates(subset=['DateTime', 'Location', 'Distance'], keep='first', inplace=True)
    df.sort_values(by=['DateTime', 'Location', 'Distance'], inplace=True)
    
    print(f"\nSuccessfully extracted {len(df)} total model data points.")
    return df


def load_measurement_data(data_directory, measurement_csv_name):
    """
    Loads and processes the complex measurement CSV file into a clean,
    long-format DataFrame.
    
    --- NEW ---
    This function is now robust and can parse headers from
    all known measurement files.
    It maps the two south sensors to 'SOUTH_1' and 'SOUTH_2'.
    """
    print(f"\n--- Loading measurement file: {measurement_csv_name} ---")
    
    file_path = measurement_csv_name # Assume full path is given
    if not os.path.exists(file_path):
        print(f"Error: Measurement file not found at '{file_path}'.")
        return None
        
    try:
        # Try to detect delimiter and settings
        # Let's try utf-8-sig first, with comma
        try:
            df_meas = pd.read_csv(
                file_path,
                delimiter=',',
                decimal='.',
                encoding='utf-8-sig' # Handles BOM
            )
            # Check if parsing worked
            if len(df_meas.columns) < 2:
                raise ValueError("Only one column found, try semicolon.")
        except Exception:
            # Fallback to semicolon delimiter
            df_meas = pd.read_csv(
                file_path,
                delimiter=';',
                decimal=',',
                encoding='utf-8-sig' # Handles BOM
            )
            
    except Exception as e:
        print(f"Error reading measurement CSV: {e}")
        return None

    try:
        # --- DYNAMIC HEADER PARSING ---
        
        # 1. Find the DateTime column
        dt_col_name = None
        if 'Date and time' in df_meas.columns:
            dt_col_name = 'Date and time'
        elif 'DateTime' in df_meas.columns:
            dt_col_name = 'DateTime'
        
        if dt_col_name is None:
            print("Error: Could not find 'Date and time' column in measurement file.")
            print(f"Columns found: {list(df_meas.columns)}")
            return None
            
        df_meas.rename(columns={dt_col_name: 'DateTime'}, inplace=True)
        
        # Try to parse date, handle multiple formats
        try:
            df_meas['DateTime'] = pd.to_datetime(df_meas['DateTime'], format='%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                df_meas['DateTime'] = pd.to_datetime(df_meas['DateTime'], format='%Y.%m.%d %H:%M:%S')
            except Exception as e:
                print(f"Error parsing DateTime, unhandled format: {e}")
                return None
        
        df_meas.set_index('DateTime', inplace=True)

        # 2. "Melt" the DataFrame from wide to long format
        df_long = df_meas.stack().reset_index()
        df_long.columns = ['DateTime', 'Header', 'Temperature']
        
        # 3. Define all possible header formats
        # We now map them to 'NORTH', 'ROOF', 'SOUTH_1', and 'SOUTH_2'
        
        # --- FIX ---
        # This regex is now more flexible and handles all known formats
        # It looks for (NORTH|SOUTH|ROOF), (ventilation string), (dist), (unit)
        header_re = re.compile(
            # TAir North Fac 50 cm [°C]
            # air temperature south facade 5 cm [C]
            # air temperature south facade close to ventilation 50 cm [C]
            r'.*?(NORTH|SOUTH|ROOF)\s*(?:Fac|facade)?(.*?)\s*(\d+)\s*(cm|m)\s*\[.*',
            re.IGNORECASE
        )
        
        parsed_header = df_long['Header'].str.extract(header_re)
        parsed_header.columns = ['Location_Str', 'Vent_Str', 'Dist_Value', 'Unit']

        # Combine back
        df_clean = pd.concat([df_long, parsed_header], axis=1)

        # 4. Standardize Location and Distance
        
        # Map location strings to clean location names
        def map_location(row):
            loc = str(row['Location_Str']).upper()
            vent = str(row['Vent_Str']).upper()
            if loc == 'NORTH':
                return 'NORTH'
            if loc == 'ROOF':
                return 'ROOF'
            if loc == 'SOUTH':
                if 'VENTILATION' in vent:
                    return 'SOUTH_2' # This is the "ventilation" sensor
                else:
                    return 'SOUTH_1' # This is the "normal" south sensor
            return None
            
        df_clean['Location'] = df_clean.apply(map_location, axis=1)
        
        # Convert types
        df_clean['Dist_Value'] = pd.to_numeric(df_clean['Dist_Value'])
        
        # Standardize distance to meters
        # --- FIX for FutureWarning ---
        df_clean['Distance'] = df_clean['Dist_Value'].astype(float) # Ensure float
        df_clean.loc[df_clean['Unit'].str.lower() == 'cm', 'Distance'] /= 100
        
        # 5. Finalize DataFrame
        final_cols = ['DateTime', 'Location', 'Distance', 'Temperature']
        df_final = df_clean[final_cols].copy()
        
        # Drop any rows where parsing failed
        df_final.dropna(inplace=True)
        
        # Convert Temperature to numeric, handling ',' decimals
        df_final['Temperature'] = df_final['Temperature'].astype(str).str.replace(',', '.')
        df_final['Temperature'] = pd.to_numeric(df_final['Temperature'], errors='coerce')
        df_final.dropna(subset=['Temperature'], inplace=True)
        
        if df_final.empty:
             print("Warning: Loaded 0 measurement data points. Check file headers/format.")
        else:
            print(f"Successfully loaded and processed {len(df_final)} measurement data points.")
            print(f"  Found locations: {list(df_final['Location'].unique())}")
            
        return df_final
        
    except Exception as e:
        print(f"An error occurred processing measurement data: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_closest_value_idx(series, target):
    """
    Helper to find the *index* (the label) of the closest value in a series to a target.
    Returns index label, or np.nan if series is empty.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if series.empty:
        return np.nan
    # Find the index in the *original* series of the minimum value
    # .idxmin() returns the *label* (which is the index)
    return (series - target).abs().idxmin()

def get_rmse(y_true, y_pred):
    """Helper to calculate RMSE, returns np.inf if error"""
    try:
        if len(y_true) < 2: return np.inf
        return np.sqrt(mean_squared_error(y_true, y_pred))
    except Exception:
        return np.inf

def get_stats_df(y_true_series, y_pred_series):
    """Aligns two time series and calculates stats"""
    if y_true_series.empty or y_pred_series.empty:
        return pd.DataFrame(columns=['Meas_Temp', 'Model_Temp'])
        
    df_true_stats = y_true_series.to_frame().reset_index().sort_values(by='DateTime')
    df_pred_stats = y_pred_series.to_frame().reset_index().sort_values(by='DateTime')
    
    df_true_stats = df_true_stats.rename(columns={'Temperature': 'Meas_Temp'})
    df_pred_stats = df_pred_stats.rename(columns={'Temperature': 'Model_Temp'})

    df_merged_stats = pd.merge_asof(
        df_true_stats,
        df_pred_stats,
        on='DateTime',
        tolerance=pd.Timedelta('30min'), # Increased tolerance for 1h data
        direction='nearest'
    )
    df_merged_stats.dropna(inplace=True)
    return df_merged_stats


def generate_timeseries_plots(df_model, df_meas, output_dir):
    """
    Generates time-series plots.
    
    --- NEW ---
    - Finds *nearest* (not best) distance match for ROOF.
    - Dynamically finds *best fit* for SOUTH1/SOUTH2 vs SOUTH_1/SOUTH_2.
    - If only SOUTH_1 is available, maps both SOUTH1/SOUTH2 to it.
    """
    print("\n--- Generating Closest-Distance Time-Series Plots ---")
    
    # --- FIX: All plots go directly into output_dir ---
    plot_dir = output_dir
    if not os.path.exists(plot_dir):
        try:
            os.makedirs(plot_dir)
            print(f"Created plot directory: {plot_dir}")
        except Exception as e:
            print(f"CRITICAL: Could not create plot directory: {e}")
            return
        
    # Get available distances for all locations
    model_dists = {loc: sorted(df_model[df_model['Location'] == loc]['Distance'].unique())
                   for loc in df_model['Location'].unique()}
    meas_dists = {loc: sorted(df_meas[df_meas['Location'] == loc]['Distance'].unique())
                  for loc in df_meas['Location'].unique()}

    # --- Safety check ---
    if not model_dists:
        print("CRITICAL: No model data was loaded. Skipping time-series plots.")
        return
    if not meas_dists:
        print("CRITICAL: No measurement data was loaded. Skipping time-series plots.")
        return

    # --- NEW DYNAMIC Comparison Map ---
    comparison_map = {}
    
    # Create a Series for easy searching
    meas_dists_series = {}
    for loc, dists in meas_dists.items():
        meas_dists_series[loc] = pd.Series(dists, index=dists)
        
    model_dists_series = {}
    for loc, dists in model_dists.items():
        model_dists_series[loc] = pd.Series(dists, index=dists)

    # 1. NORTH (Find closest match)
    if 'NORTH' in model_dists_series and 'NORTH' in meas_dists_series:
        model_dist_n = model_dists['NORTH'][0] # Closest model dist to wall
        meas_idx_n = find_closest_value_idx(meas_dists_series['NORTH'], model_dist_n)
        if pd.notna(meas_idx_n):
            meas_dist_n = meas_dists_series['NORTH'][meas_idx_n]
            comparison_map['NORTH'] = {'model_loc': 'NORTH', 'model_dist': model_dist_n,
                                       'meas_loc': 'NORTH', 'meas_dist': meas_dist_n}
    
    # 2. ROOF (Nearest Height Logic)
    if 'ROOF' in model_dists_series and 'ROOF' in meas_dists_series:
        # Find meas dist closest to 1.0m (a common, representative height)
        target_meas_dist = 1.0
        meas_idx_r = find_closest_value_idx(meas_dists_series['ROOF'], target_meas_dist)
        if pd.notna(meas_idx_r):
             meas_dist_r = meas_dists_series['ROOF'][meas_idx_r] # e.g., 1.0m
             # Now find the model dist *nearest to that measurement dist*
             model_idx_r = find_closest_value_idx(model_dists_series['ROOF'], meas_dist_r)
             if pd.notna(model_idx_r):
                model_dist_r = model_dists_series['ROOF'][model_idx_r] # e.g., 1.25m
                
                comparison_map['ROOF'] = {'model_loc': 'ROOF', 'model_dist': model_dist_r,
                                          'meas_loc': 'ROOF', 'meas_dist': meas_dist_r}

    # 3. SOUTH (Dynamic "Best Fit Swap" Logic)
    meas_has_south1 = 'SOUTH_1' in meas_dists_series
    meas_has_south2 = 'SOUTH_2' in meas_dists_series
    model_has_south1 = 'SOUTH1' in model_dists_series
    model_has_south2 = 'SOUTH2' in model_dists_series

    # Case 1: LONG file (both meas sensors exist)
    if meas_has_south1 and meas_has_south2 and model_has_south1 and model_has_south2:
        print("Info: Found two SOUTH measurement sensors. Calculating best fit...")
        
        # Get the series for all 4
        m1_dist = model_dists['SOUTH1'][0]
        m2_dist = model_dists['SOUTH2'][0]
        # Find closest meas dist for each model dist
        s1_dist_idx = find_closest_value_idx(meas_dists_series['SOUTH_1'], m1_dist)
        s2_dist_idx = find_closest_value_idx(meas_dists_series['SOUTH_2'], m2_dist)
        
        s1_dist = meas_dists_series['SOUTH_1'][s1_dist_idx]
        s2_dist = meas_dists_series['SOUTH_2'][s2_dist_idx]

        s_m1 = df_model[(df_model['Location'] == 'SOUTH1') & (abs(df_model['Distance'] - m1_dist) < 0.001)].set_index('DateTime')['Temperature'].dropna()
        s_m2 = df_model[(df_model['Location'] == 'SOUTH2') & (abs(df_model['Distance'] - m2_dist) < 0.001)].set_index('DateTime')['Temperature'].dropna()
        s_s1 = df_meas[(df_meas['Location'] == 'SOUTH_1') & (abs(df_meas['Distance'] - s1_dist) < 0.001)].set_index('DateTime')['Temperature'].dropna()
        s_s2 = df_meas[(df_meas['Location'] == 'SOUTH_2') & (abs(df_meas['Distance'] - s2_dist) < 0.001)].set_index('DateTime')['Temperature'].dropna()

        # Align and get RMSE for all 4 combinations
        rmse_11 = get_rmse(get_stats_df(s_s1, s_m1)['Meas_Temp'], get_stats_df(s_s1, s_m1)['Model_Temp'])
        rmse_12 = get_rmse(get_stats_df(s_s1, s_m2)['Meas_Temp'], get_stats_df(s_s1, s_m2)['Model_Temp'])
        rmse_21 = get_rmse(get_stats_df(s_s2, s_m1)['Meas_Temp'], get_stats_df(s_s2, s_m1)['Model_Temp'])
        rmse_22 = get_rmse(get_stats_df(s_s2, s_m2)['Meas_Temp'], get_stats_df(s_s2, s_m2)['Model_Temp'])
        
        # Check which total error is lower
        if (rmse_11 + rmse_22) <= (rmse_12 + rmse_21):
            print("Info: Best fit is S1->S_1 and S2->S_2. (No swap)")
            comparison_map['SOUTH1'] = {'model_loc': 'SOUTH1', 'model_dist': m1_dist, 'meas_loc': 'SOUTH_1', 'meas_dist': s1_dist}
            comparison_map['SOUTH2'] = {'model_loc': 'SOUTH2', 'model_dist': m2_dist, 'meas_loc': 'SOUTH_2', 'meas_dist': s2_dist}
        else:
            print("Info: Best fit is S1->S_2 and S2->S_1. (SWAPPED)")
            comparison_map['SOUTH1_SWAPPED'] = {'model_loc': 'SOUTH1', 'model_dist': m1_dist, 'meas_loc': 'SOUTH_2', 'meas_dist': s2_dist}
            comparison_map['SOUTH2_SWAPPED'] = {'model_loc': 'SOUTH2', 'model_dist': m2_dist, 'meas_loc': 'SOUTH_1', 'meas_dist': s1_dist}

    # Case 2: SHORT file (only one meas sensor exists)
    elif meas_has_south1 and (model_has_south1 or model_has_south2):
        print("Info: Found one SOUTH measurement sensor. Comparing both S1 and S2 models to it.")
        if model_has_south1:
            m1_dist = model_dists['SOUTH1'][0]
            s1_dist_idx = find_closest_value_idx(meas_dists_series['SOUTH_1'], m1_dist)
            s1_dist = meas_dists_series['SOUTH_1'][s1_dist_idx]
            comparison_map['SOUTH1_vs_S1'] = {'model_loc': 'SOUTH1', 'model_dist': m1_dist, 'meas_loc': 'SOUTH_1', 'meas_dist': s1_dist}
        if model_has_south2:
            m2_dist = model_dists['SOUTH2'][0]
            s1_dist_idx = find_closest_value_idx(meas_dists_series['SOUTH_1'], m2_dist)
            s1_dist = meas_dists_series['SOUTH_1'][s1_dist_idx]
            comparison_map['SOUTH2_vs_S1'] = {'model_loc': 'SOUTH2', 'model_dist': m2_dist, 'meas_loc': 'SOUTH_1', 'meas_dist': s1_dist}


    print("\n--- Plotting Map (Model vs. Measurement) ---")
    if not comparison_map:
        print("Warning: No valid comparison pairs were found.")
    else:
        for key, val in comparison_map.items():
            print(f"Plot '{key}': Model({val['model_loc']} @ {val['model_dist']:.2f}m) vs. Meas({val['meas_loc']} @ {val['meas_dist']:.2f}m)")
    print("-----------------------------------------------")

    # Now, create a plot for each item in our map
    for plot_name, p in comparison_map.items():
        
        try:
            # Get the two data series we want to compare
            y_pred_series = df_model[
                (df_model['Location'] == p['model_loc']) &
                (abs(df_model['Distance'] - p['model_dist']) < 0.001)
            ].set_index('DateTime')['Temperature'].dropna()
            
            y_true_series = df_meas[
                (df_meas['Location'] == p['meas_loc']) &
                (abs(df_meas['Distance'] - p['meas_dist']) < 0.001)
            ].set_index('DateTime')['Temperature'].dropna()

            if y_true_series.empty or y_pred_series.empty:
                print(f"Warning: No data for plot '{plot_name}'. Skipping.")
                continue
                
            # --- Align data for STATS (using merge_asof) ---
            df_merged_stats = get_stats_df(y_true_series, y_pred_series)

            info_text = "No overlapping data for stats"
            if not df_merged_stats.empty and len(df_merged_stats) > 2:
                
                y_true_stats = df_merged_stats['Meas_Temp']
                y_pred_stats = df_merged_stats['Model_Temp']

                # R-squared (Coefficient of Determination)
                r2_det = r2_score(y_true_stats, y_pred_stats)
                # R-squared (Pearson Correlation)
                corr, _ = pearsonr(y_true_stats, y_pred_stats)
                r2_corr = corr**2
                
                rmse = np.sqrt(mean_squared_error(y_true_stats, y_pred_stats))
                
                info_text = f"R² (det.): {r2_det:.3f}\nr² (corr.): {r2_corr:.3f}\nRMSE: {rmse:.3f} °C"
                print(f"Stats for {plot_name}: R²={r2_det:.3f}, r²={r2_corr:.3f}, RMSE={rmse:.3f} °C")
            else:
                print(f"CRITICAL WARNING: No overlapping data found for '{plot_name}' within 30min tolerance.")

            # --- Create Plot (using the raw, un-merged data) ---
            plt.figure(figsize=(15, 7))
            
            model_label = f"Model ({p['model_loc']} @ {p['model_dist']:.2f}m)"
            meas_label = f"Meas. ({p['meas_loc']} @ {p['meas_dist']:.2f}m)"
            
            plt.plot(y_true_series.index, y_true_series,
                     label=meas_label,
                     color='black', alpha=0.9, marker='.', markersize=4, linestyle='-')
            plt.plot(y_pred_series.index, y_pred_series,
                     label=model_label,
                     color='red', alpha=0.8, marker='.', markersize=4, linestyle='-')

            plt.title(f"Model vs. Measurement: {plot_name}", fontsize=16)
            plt.xlabel('Date / Time', fontsize=12)
            plt.ylabel('Air Temperature (°C)', fontsize=12)
            plt.legend()
            plt.grid(True, which='both', linestyle='--', alpha=0.6)

            # Set X-axis to overlapping range
            start_time = max(y_true_series.index.min(), y_pred_series.index.min())
            end_time = min(y_true_series.index.max(), y_pred_series.index.max())

            if start_time < end_time:
                plt.xlim(start_time, end_time)

            # Add the stats box
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                     verticalalignment='top', fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            
            # --- FIX: Save to main output_dir ---
            plot_filename = f'timeseries_comparison_{plot_name}.png'
            plot_filepath = os.path.join(plot_dir, plot_filename)
            plt.savefig(plot_filepath)
            plt.close()
            
            print(f"Successfully created plot: {plot_filename}")

        except Exception as e:
            print(f"An error occurred during time-series plot generation for {plot_name}: {e}")
            import traceback
            traceback.print_exc()


def get_profile_at_time(df, location, time_str):
    """
    Helper function to get a temperature profile (Temp vs. Distance)
    for a specific location and time.
    """
    try:
        # Filter by location first
        df_loc = df[df['Location'] == location]
        if df_loc.empty:
            return pd.DataFrame(columns=['DateTime', 'Location', 'Distance', 'Temperature'])

        # Find the *nearest* available timestamp within a tolerance
        target_time = pd.to_datetime(time_str)
        
        # Get all unique datetimes for this location
        unique_times = df_loc['DateTime'].unique()
        if len(unique_times) == 0:
            return pd.DataFrame(columns=['DateTime', 'Location', 'Distance', 'Temperature'])
            
        # Find the absolute difference in time
        time_diff = pd.Series(unique_times) - target_time
        
        # Find the index of the minimum difference
        closest_time_idx = time_diff.abs().idxmin()
        
        # Get the actual closest time
        closest_time = unique_times[closest_time_idx]

        # Check if the closest time is within our tolerance (e.g., 30 minutes for 1h data)
        if abs(closest_time - target_time) <= pd.Timedelta('30 minutes'):
            # Return all rows matching that *exact* closest time
            profile_data = df_loc[df_loc['DateTime'] == closest_time]
            return profile_data.sort_values(by='Distance')
        else:
            print(f"Could not find data for {location} at {time_str} (no match within 30 min)")
            return pd.DataFrame(columns=['DateTime', 'Location', 'Distance', 'Temperature'])
            
    except Exception as e:
        print(f"Could not find data for {location} at {time_str}: {e}")
        return pd.DataFrame(columns=['DateTime', 'Location', 'Distance', 'Temperature'])


def generate_gradient_plots(df_model, df_meas, output_dir):
    """
    Generates gradient profile plots (Temp vs. Distance) for
    specific locations and times.
    
    --- NEW ---
    - Creates a *separate plot for each time*.
    - Saves all plots to the main `output_dir`.
    """
    print("\n--- Generating Gradient Profile Plots ---")
    
    # --- FIX: All plots go directly into output_dir ---
    plot_dir = output_dir
    if not os.path.exists(plot_dir):
        try:
            os.makedirs(plot_dir)
            print(f"Created plot directory: {plot_dir}")
        except Exception as e:
            print(f"CRITICAL: Could not create plot directory: {e}")
            return
        
    # Timesteps to plot (as strings)
    # Get a default time if data is available
    
    default_time = None
    if df_model is not None and not df_model.empty:
        default_time = df_model['DateTime'].median().strftime('%Y-%m-%d %H:%M:%S')
    elif df_meas is not None and not df_meas.empty:
        default_time = df_meas['DateTime'].median().strftime('%Y-%m-%d %H:%M:%S')
    else:
        default_time = '2022-07-19 13:00:00' # Fallback
        
    # Try to find a good day peak
    peak_time = default_time
    try:
        # Find peak time from measurements if available
        if df_meas is not None and not df_meas.empty:
            peak_time_dt = df_meas.loc[df_meas['Temperature'].idxmax()]['DateTime']
            peak_time = peak_time_dt.replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
        elif df_model is not None and not df_model.empty:
            peak_time_dt = df_model.loc[df_model['Temperature'].idxmax()]['DateTime']
            peak_time = peak_time_dt.replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
         pass # Keep default time if peak finding fails

    times_to_plot = [
        (pd.to_datetime(peak_time) - pd.Timedelta('3 hours')).strftime('%Y-%m-%d %H:%M:%S'),
        peak_time,
        (pd.to_datetime(peak_time) + pd.Timedelta('3 hours')).strftime('%Y-%m-%d %H:%M:%S'),
        (pd.to_datetime(peak_time) + pd.Timedelta('9 hours')).strftime('%Y-%m-%d %H:%M:%S') # Night
    ]
    
    print(f"Plotting gradients for times: {times_to_plot}")
    
    # Get all available locations
    all_model_locs = df_model['Location'].unique()
    all_meas_locs = df_meas['Location'].unique()

    for plot_loc in ['ROOF', 'NORTH', 'SOUTH']:
        
        # --- NEW: Create a separate plot for each time ---
        for i, time_str in enumerate(times_to_plot):
            plt.figure(figsize=(10, 7))
            has_data = False
            time_label = time_str[11:16].replace(":", "") # e.g., "1000"
            
            # --- Get Measurement Data ---
            if plot_loc == 'SOUTH':
                # Plot both SOUTH_1 and SOUTH_2 if they exist
                if 'SOUTH_1' in all_meas_locs:
                    df_meas_s1 = get_profile_at_time(df_meas, 'SOUTH_1', time_str)
                    if not df_meas_s1.empty:
                        plt.plot(df_meas_s1['Distance'], df_meas_s1['Temperature'],
                                 label=f"Meas. S1",
                                 color='black', linestyle='--', marker='o')
                        has_data = True
                if 'SOUTH_2' in all_meas_locs:
                    df_meas_s2 = get_profile_at_time(df_meas, 'SOUTH_2', time_str)
                    if not df_meas_s2.empty:
                        plt.plot(df_meas_s2['Distance'], df_meas_s2['Temperature'],
                                 label=f"Meas. S2",
                                 color='grey', linestyle=':', marker='v')
                        has_data = True
            else:
                 if plot_loc in all_meas_locs:
                     df_meas_profile = get_profile_at_time(df_meas, plot_loc, time_str)
                     if not df_meas_profile.empty:
                        plt.plot(df_meas_profile['Distance'], df_meas_profile['Temperature'],
                                 label=f"Meas.",
                                 color='black', linestyle='--', marker='o')
                        has_data = True

            # --- Get Model Data ---
            if plot_loc == 'SOUTH':
                # Plot SOUTH1, SOUTH2, and SOUTH if they exist
                if 'SOUTH1' in all_model_locs:
                    df_model_s1 = get_profile_at_time(df_model, 'SOUTH1', time_str)
                    if not df_model_s1.empty:
                        plt.plot(df_model_s1['Distance'], df_model_s1['Temperature'],
                                 label=f"Model S1",
                                 color='red', linestyle='-', marker='x')
                        has_data = True
                if 'SOUTH2' in all_model_locs:
                    df_model_s2 = get_profile_at_time(df_model, 'SOUTH2', time_str)
                    if not df_model_s2.empty:
                        plt.plot(df_model_s2['Distance'], df_model_s2['Temperature'],
                                 label=f"Model S2",
                                 color='blue', linestyle='-', marker='+')
                        has_data = True
                if 'SOUTH' in all_model_locs:
                    df_model_s = get_profile_at_time(df_model, 'SOUTH', time_str)
                    if not df_model_s.empty:
                        plt.plot(df_model_s['Distance'], df_model_s['Temperature'],
                                 label=f"Model S",
                                 color='green', linestyle='-', marker='*')
                        has_data = True
            else:
                if plot_loc in all_model_locs:
                    df_model_profile = get_profile_at_time(df_model, plot_loc, time_str)
                    if not df_model_profile.empty:
                        plt.plot(df_model_profile['Distance'], df_model_profile['Temperature'],
                                 label=f"Model",
                                 color='red', linestyle='-', marker='x')
                        has_data = True
            
            if not has_data:
                # print(f"No data found for gradient plot: {plot_loc} @ {time_label}")
                plt.close() # Don't save an empty plot
                continue # Skip to next time
            
            plt.title(f'Temperature Gradient: {plot_loc} at {time_str}')
            plt.xlabel('Distance from Surface (m)')
            plt.ylabel('Air Temperature (°C)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='best', fontsize='small')
            plt.tight_layout()
            
            plot_filename = f'gradient_profile_{plot_loc}_{time_label}.png'
            plot_filepath = os.path.join(plot_dir, plot_filename)
            plt.savefig(plot_filepath)
            plt.close()
            
            print(f"Successfully created plot: {plot_filename}")
        
    print("\n--- All processing complete ---")


if __name__ == '__main__':
    # --- User-defined section ---
    # !! IMPORTANT: Update this path to your data folder !!
    # This script will search this folder AND all subfolders
    # Use r'...' (raw string) to avoid SyntaxWarning on Windows
    #data_directory = r'Y:\Danmark_Building\Danmark_Building_Validation_Terrain_Short\receptors'
    data_directory = r'Y:\Danmark_Building\Danmark_Building_Validation_Long_MO\receptors'

    # Use r'...' for all Windows paths
    #measurement_csv_name = r'D:\enviprojects\Projektwerkstatt_Nissen_Schoefl\Measurements_220719_1hCorr_ALLDISTS.csv' 
    measurement_csv_name = r'D:\enviprojects\Projektwerkstatt_Nissen_Schoefl\Measurements_LongPeriod_1hCorr_ALLDISTS.csv' 
    #measurement_csv_name = r'D:\enviprojects\Projektwerkstatt_Nissen_Schoefl\Measurements_220719_1hCorr.csv' 
    #measurement_csv_name = r'D:\enviprojects\Projektwerkstatt_Nissen_Schoefl\Measurements_LongPeriod_1hCorr.csv' 
    output_dir = r'D:\CompPlotsDanmark'
    # --- End of User-defined section ---

    if not os.path.isdir(data_directory):
        print(f"CRITICAL ERROR: The specified directory does not exist:\n{data_directory}")
        exit()
    if not os.path.isfile(measurement_csv_name):
        print(f"CRITICAL ERROR: The measurement file does not exist:\n{measurement_csv_name}")
        exit()
    if not os.path.isdir(output_dir):
        print(f"Warning: Output directory does not exist. Attempting to create it.")
        try:
            os.makedirs(output_dir)
            print(f"Successfully created output directory: {output_dir}")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not create output directory: {e}")
            exit()

    # --- 1. Extract all model data ---
    df_model = extract_model_data(data_directory)

    # --- 2. Load all measurement data ---
    df_meas = load_measurement_data(data_directory, measurement_csv_name)

    # --- 3. Generate plots (if both steps were successful) ---
    if (df_model is not None and not df_model.empty) or \
       (df_meas is not None and not df_meas.empty):
        
        # Check if dataframes are valid
        if df_model is None or df_model.empty:
            print("Warning: Model data is empty. Skipping plots.")
            df_model = pd.DataFrame(columns=['DateTime', 'Location', 'Distance', 'Temperature']) # Create empty for safety
        if df_meas is None or df_meas.empty:
            print("Warning: Measurement data is empty. Skipping plots.")
            df_meas = pd.DataFrame(columns=['DateTime', 'Location', 'Distance', 'Temperature']) # Create empty for safety
            
        # Generate the new "closest distance" time-series plots
        generate_timeseries_plots(df_model.copy(), df_meas.copy(), output_dir)
        
        # Generate the new "gradient profile" plots
        generate_gradient_plots(df_model.copy(), df_meas.copy(), output_dir)
    else:
        print("\nCRITICAL: No data was loaded from model or measurements. Skipping plot generation.")