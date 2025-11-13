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
    
    # --- FIX ---
    # New Regex to find location and distance from *either* format:
    # Format 1: (NORTH)(125)(CM)
    # Format 2: (NORTH)_(05)
    # This captures the location, then *optionally* a group for _XX or a group for XXXCM
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
                            current_z = float(parts[z_index])
                            # Convert absolute Z height to distance above roof
                            distance = current_z - roof_base_height
                            
                            # Skip any data from *below* the roof surface
                            if distance < -0.01: # Small tolerance
                                continue
                                
                            temp = float(parts[temp_index])
                            dt_val = parts[dt_index]
                            
                            all_data_list.append([dt_val, location, distance, temp])
                            
                        except (ValueError, IndexError):
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
                        # This might be a file like 'SOUTH1.AT_1DT' without distance
                        # Let's assume a default distance or z-height
                        print(f"Warning: No distance in name for {filename}. Defaulting to z=1.75m.")
                        distance_m = 1.75 # Default distance if not specified
                    
                    # For all facade files, extract data at the standard
                    # 1.75m height above ground.
                    target_z_height = 1.75

                    for line in f:
                        if not line.strip(): continue
                        parts = [p.strip() for p in line.strip().split(',')]
                        if len(parts) <= max(z_index, temp_index, dt_index): continue
                        
                        try:
                            current_z = float(parts[z_index])
                            
                            # We only want data from the standard 1.75m height
                            if abs(current_z - target_z_height) < 0.01:
                                temp = float(parts[temp_index])
                                dt_val = parts[dt_index]
                                # Save with the distance from the FILENAME
                                all_data_list.append([dt_val, location, distance_m, temp])

                        except (ValueError, IndexError):
                            continue
                            
        except FileNotFoundError:
            print(f"Error: File not found at '{file_path}'")
        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e}")

    if not all_data_list:
        print("CRITICAL ERROR: No model data was successfully extracted.")
        return None

    # Convert list to DataFrame
    df = pd.DataFrame(all_data_list, columns=['DateTime', 'Location', 'Distance', 'Temperature'])
    
    # Convert types
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
    df['Distance'] = pd.to_numeric(df['Distance'])
    df['Temperature'] = pd.to_numeric(df['Temperature'])
    
    # Clean up data
    df.drop_duplicates(inplace=True)
    df.sort_values(by=['DateTime', 'Location', 'Distance'], inplace=True)
    
    print(f"\nSuccessfully extracted {len(df)} total model data points.")
    return df


def load_measurement_data(data_directory, measurement_csv_name):
    """
    Loads and processes the complex measurement CSV file into a clean,
    long-format DataFrame.
    
    --- NEW ---
    This function is now robust and can parse headers from
    'Measurements_220719_1hCorr.csv' AND '...ALLDISTS.csv'
    """
    print(f"\n--- Loading measurement file: {measurement_csv_name} ---")
    
    file_path = measurement_csv_name # Assume full path is given
    if not os.path.exists(file_path):
        print(f"Error: Measurement file not found at '{file_path}'.")
        return None
        
    try:
        # Try to detect delimiter and settings
        # Let's try utf-8-sig first
        try:
            df_meas = pd.read_csv(
                file_path,
                delimiter=',',
                decimal='.',
                encoding='utf-8-sig' # Handles BOM
            )
        except Exception:
            # Fallback to semicolon delimiter
            df_meas = pd.read_csv(
                file_path,
                delimiter=';',
                decimal=',',
                encoding='utf-8-sig'
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
        # Format 1: "air temperature north facade 5 cm [C]" (ALLDISTS file)
        re_format1 = re.compile(
            r'air temperature (north|south|roof) facade (\d+) (cm|m)\s*\[C\]',
            re.IGNORECASE
        )
        # Format 2: "TAir South Fac 50 cm [°C]" (1hCorr file)
        re_format2 = re.compile(
            r'TAir (North|South) Fac (\d+) cm \[°C\]',
            re.IGNORECASE
        )
        # Format 3: "Tair Roof 100 cm [°C]" (1hCorr file)
        re_format3 = re.compile(
            r'Tair (Roof) (\d+) cm \[°C\]',
            re.IGNORECASE
        )

        # Apply regexes and combine results
        parsed1 = df_long['Header'].str.extract(re_format1)
        parsed1.columns = ['Location', 'Dist_Value', 'Unit']
        
        parsed2 = df_long['Header'].str.extract(re_format2)
        parsed2.columns = ['Location', 'Dist_Value']
        parsed2['Unit'] = 'cm' # This format is always cm
        
        parsed3 = df_long['Header'].str.extract(re_format3)
        parsed3.columns = ['Location', 'Dist_Value']
        parsed3['Unit'] = 'cm' # This format is always cm

        # Combine parsed data, format 1 takes precedence
        parsed_header = parsed1.fillna(parsed2).fillna(parsed3)
        
        # Combine back
        df_clean = pd.concat([df_long, parsed_header], axis=1)
        
        # Convert types
        df_clean['Dist_Value'] = pd.to_numeric(df_clean['Dist_Value'])
        
        # Standardize distance to meters
        df_clean['Distance'] = df_clean['Dist_Value']
        df_clean.loc[df_clean['Unit'].str.lower() == 'cm', 'Distance'] /= 100
        
        # Standardize location names
        df_clean['Location'] = df_clean['Location'].str.upper()
        
        # Keep only the columns we need
        final_cols = ['DateTime', 'Location', 'Distance', 'Temperature']
        df_final = df_clean[final_cols].copy()
        
        # Drop any rows where parsing failed
        df_final.dropna(inplace=True)
        
        # Convert Temperature to numeric, handling ',' decimals
        df_final['Temperature'] = df_final['Temperature'].astype(str).str.replace(',', '.')
        df_final['Temperature'] = pd.to_numeric(df_final['Temperature'], errors='coerce')
        df_final.dropna(subset=['Temperature'], inplace=True)

        print(f"Successfully loaded and processed {len(df_final)} measurement data points.")
        return df_final
        
    except Exception as e:
        print(f"An error occurred processing measurement data: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_closest_value(series, target):
    """Helper to find the closest value in a series to a target."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if series.empty:
        return np.nan
    return (series - target).abs().idxmin()


def generate_timeseries_plots(df_model, df_meas, output_dir):
    """
    Generates time-series plots comparing the *closest available distance*
    for each location (e.g., Model @ 0.25m vs Meas @ 0.2m).
    """
    print("\n--- Generating Closest-Distance Time-Series Plots ---")
    
    plot_dir = os.path.join(output_dir, 'comparison_plots_timeseries')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created plot directory: {plot_dir}")
        
    # Get all unique locations from both dataframes
    all_locations = set(df_model['Location'].unique()) | set(df_meas['Location'].unique())
    # We may need to map SOUTH1/SOUTH2 to SOUTH
    
    # --- Define our "Closest" comparison map ---
    # This is complex. We'll map model locations to measurement locations
    comparison_map = {}
    
    # Get available distances for all locations
    model_dists = {loc: sorted(df_model[df_model['Location'] == loc]['Distance'].unique())
                   for loc in df_model['Location'].unique()}
    meas_dists = {loc: sorted(df_meas[df_meas['Location'] == loc]['Distance'].unique())
                  for loc in df_meas['Location'].unique()}

    # --- Safety check ---
    if not model_dists or not meas_dists:
        print("CRITICAL: No data found in model or measurement dataframes. Skipping plots.")
        return

    # 1. NORTH
    if 'NORTH' in model_dists and 'NORTH' in meas_dists and model_dists['NORTH'] and meas_dists['NORTH']:
        model_dist_n = model_dists['NORTH'][0] # Closest model dist
        meas_idx_n = find_closest_value(pd.Series(meas_dists['NORTH']), model_dist_n)
        if pd.notna(meas_idx_n):
            meas_dist_n = meas_dists['NORTH'][meas_idx_n]
            comparison_map['NORTH'] = {'model_loc': 'NORTH', 'model_dist': model_dist_n,
                                       'meas_loc': 'NORTH', 'meas_dist': meas_dist_n}
    
    # 2. ROOF
    if 'ROOF' in model_dists and 'ROOF' in meas_dists and model_dists['ROOF'] and meas_dists['ROOF']:
        model_dist_r = model_dists['ROOF'][0] # Closest model dist (e.g., 0.25m)
        meas_idx_r = find_closest_value(pd.Series(meas_dists['ROOF']), model_dist_r)
        if pd.notna(meas_idx_r):
            meas_dist_r = meas_dists['ROOF'][meas_idx_r]
            comparison_map['ROOF'] = {'model_loc': 'ROOF', 'model_dist': model_dist_r,
                                      'meas_loc': 'ROOF', 'meas_dist': meas_dist_r}

    # 3. SOUTH1
    if 'SOUTH1' in model_dists and 'SOUTH' in meas_dists and model_dists['SOUTH1'] and meas_dists['SOUTH']:
        model_dist_s1 = model_dists['SOUTH1'][0]
        meas_idx_s1 = find_closest_value(pd.Series(meas_dists['SOUTH']), model_dist_s1)
        if pd.notna(meas_idx_s1):
            meas_dist_s1 = meas_dists['SOUTH'][meas_idx_s1]
            comparison_map['SOUTH1'] = {'model_loc': 'SOUTH1', 'model_dist': model_dist_s1,
                                        'meas_loc': 'SOUTH', 'meas_dist': meas_dist_s1}
                                    
    # 4. SOUTH2
    if 'SOUTH2' in model_dists and 'SOUTH' in meas_dists and model_dists['SOUTH2'] and meas_dists['SOUTH']:
        model_dist_s2 = model_dists['SOUTH2'][0]
        meas_idx_s2 = find_closest_value(pd.Series(meas_dists['SOUTH']), model_dist_s2)
        if pd.notna(meas_idx_s2):
            meas_dist_s2 = meas_dists['SOUTH'][meas_idx_s2]
            comparison_map['SOUTH2'] = {'model_loc': 'SOUTH2', 'model_dist': model_dist_s2,
                                        'meas_loc': 'SOUTH', 'meas_dist': meas_dist_s2}
                                    
    # 5. SOUTH (fallback)
    if 'SOUTH' in model_dists and 'SOUTH' in meas_dists and model_dists['SOUTH'] and meas_dists['SOUTH']:
        model_dist_s = model_dists['SOUTH'][0]
        meas_idx_s = find_closest_value(pd.Series(meas_dists['SOUTH']), model_dist_s)
        if pd.notna(meas_idx_s):
            meas_dist_s = meas_dists['SOUTH'][meas_idx_s]
            comparison_map['SOUTH'] = {'model_loc': 'SOUTH', 'model_dist': model_dist_s,
                                       'meas_loc': 'SOUTH', 'meas_dist': meas_dist_s}

    print("\n--- Plotting Map (Model vs. Measurement) ---")
    if not comparison_map:
        print("Warning: No valid comparison pairs were found.")
    else:
        for key, val in comparison_map.items():
            print(f"Plot '{key}': Model({val['model_loc']} @ {val['model_dist']}m) vs. Meas({val['meas_loc']} @ {val['meas_dist']}m)")
    print("-----------------------------------------------")

    # Now, create a plot for each item in our map
    for plot_name, p in comparison_map.items():
        
        try:
            # Get the two data series we want to compare
            y_pred_series = df_model[
                (df_model['Location'] == p['model_loc']) &
                (df_model['Distance'] == p['model_dist'])
            ].set_index('DateTime')['Temperature'].dropna()
            
            y_true_series = df_meas[
                (df_meas['Location'] == p['meas_loc']) &
                (df_meas['Distance'] == p['meas_dist'])
            ].set_index('DateTime')['Temperature'].dropna()

            if y_true_series.empty or y_pred_series.empty:
                print(f"Warning: No data for plot '{plot_name}'. Skipping.")
                continue
                
            # --- Align data for STATS (using merge_asof) ---
            df_true_stats = y_true_series.to_frame().reset_index().sort_values(by='DateTime')
            df_pred_stats = y_pred_series.to_frame().reset_index().sort_values(by='DateTime')
            
            # Rename columns to avoid conflict
            df_true_stats = df_true_stats.rename(columns={'Temperature': 'Meas_Temp'})
            df_pred_stats = df_pred_stats.rename(columns={'Temperature': 'Model_Temp'})

            df_merged_stats = pd.merge_asof(
                df_true_stats,
                df_pred_stats,
                on='DateTime',
                tolerance=pd.Timedelta('10min'),
                direction='nearest'
            )
            df_merged_stats.dropna(inplace=True)

            info_text = "No overlapping data for stats"
            if not df_merged_stats.empty:
                
                # --- FIX for KeyError ---
                # Use the new, conflict-free column names
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
                print(f"CRITICAL WARNING: No overlapping data found for '{plot_name}' within 10min tolerance.")

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
    """
    print("\n--- Generating Gradient Profile Plots ---")
    
    plot_dir = os.path.join(output_dir, 'comparison_plots_gradient')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
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
    if df_model is not None and not df_model.empty:
        peak_time = df_model.loc[df_model['Temperature'].idxmax()]['DateTime'].strftime('%Y-%m-%d %H:00:00')
    elif df_meas is not None and not df_meas.empty:
        peak_time = df_meas.loc[df_meas['Temperature'].idxmax()]['DateTime'].strftime('%Y-%m-%d %H:00:00')

    times_to_plot = [
        (pd.to_datetime(peak_time) - pd.Timedelta('3 hours')).strftime('%Y-%m-%d %H:%M:%S'),
        peak_time,
        (pd.to_datetime(peak_time) + pd.Timedelta('3 hours')).strftime('%Y-%m-%d %H:%M:%S'),
        (pd.to_datetime(peak_time) + pd.Timedelta('9 hours')).strftime('%Y-%m-%d %H:%M:%S') # Night
    ]
    
    print(f"Plotting gradients for times: {times_to_plot}")
    
    # Locations to plot
    # We map Model 'SOUTH1'/'SOUTH2' to Measurement 'SOUTH'
    locations_to_plot = {
        'ROOF': 'ROOF',
        'NORTH': 'NORTH',
        'SOUTH': 'SOUTH' # This will plot Model(SOUTH1+SOUTH2+SOUTH) vs Meas(SOUTH)
    }

    for model_loc_key, meas_loc in locations_to_plot.items():
        plt.figure(figsize=(10, 7))
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(times_to_plot)))
        
        has_data = False # Flag to check if we plot anything
        
        for i, time_str in enumerate(times_to_plot):
            
            # --- Get Measurement Data ---
            df_meas_profile = get_profile_at_time(df_meas, meas_loc, time_str)
            
            # --- Get Model Data ---
            # --- FIX for KeyError ---
            # Build a list of dataframes to concat, then check if list is empty
            model_profiles_to_concat = []
            if model_loc_key == 'SOUTH':
                # Combine SOUTH1, SOUTH2, and SOUTH for the model plot
                df_model_s1 = get_profile_at_time(df_model, 'SOUTH1', time_str)
                df_model_s2 = get_profile_at_time(df_model, 'SOUTH2', time_str)
                df_model_s = get_profile_at_time(df_model, 'SOUTH', time_str)
                
                if not df_model_s1.empty:
                    model_profiles_to_concat.append(df_model_s1)
                if not df_model_s2.empty:
                    model_profiles_to_concat.append(df_model_s2)
                if not df_model_s.empty:
                    model_profiles_to_concat.append(df_model_s)
            else:
                df_model_profile_loc = get_profile_at_time(df_model, model_loc_key, time_str)
                if not df_model_profile_loc.empty:
                    model_profiles_to_concat.append(df_model_profile_loc)
            
            # Only concat and plot if we have data
            df_model_profile = pd.DataFrame() # Ensure it's defined
            if model_profiles_to_concat:
                df_model_profile = pd.concat(model_profiles_to_concat).sort_values(by='Distance')

            # Plot Measurement data
            if not df_meas_profile.empty:
                plt.plot(df_meas_profile['Distance'], df_meas_profile['Temperature'],
                         label=f"Meas. {time_str[11:16]}",
                         color=colors[i], linestyle='--', marker='o')
                has_data = True
            
            # Plot Model data
            if not df_model_profile.empty:
                plt.plot(df_model_profile['Distance'], df_model_profile['Temperature'],
                         label=f"Model {time_str[11:16]}",
                         color=colors[i], linestyle='-', marker='x')
                has_data = True
        
        if not has_data:
            print(f"No data found for gradient plot: {model_loc_key}")
            plt.close() # Don't save an empty plot
            continue # Skip to next location

        plt.title(f'Temperature Gradient: {model_loc_key}')
        plt.xlabel('Distance from Surface (m)')
        plt.ylabel('Air Temperature (°C)')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Create a clean legend
        # Get unique labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            plt.legend(by_label.values(), by_label.keys(), loc='best')

        plt.tight_layout()
        
        plot_filename = f'gradient_profile_{model_loc_key}.png'
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
    data_directory = r'Y:\Danmark_Building\Danmark_Building_Validation_Long\receptors'
    
    # Use r'...' for all Windows paths
    #measurement_csv_name = r'D:\enviprojects\Projektwerkstatt_Nissen_Schoefl\Measurements_220719_1hCorr_ALLDISTS.csv' 
    #measurement_csv_name = r'D:\enviprojects\Projektwerkstatt_Nissen_Schoefl\Measurements_220719_1hCorr.csv' 
    measurement_csv_name = r'D:\enviprojects\Projektwerkstatt_Nissen_Schoefl\Measurements_LongPeriod_1hCorr.csv' 
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
        
        # Generate the new "closest distance" time-series plots
        generate_timeseries_plots(df_model.copy(), df_meas.copy(), output_dir)
        
        # Generate the new "gradient profile" plots
        generate_gradient_plots(df_model.copy(), df_meas.copy(), output_dir)
    else:
        print("\nCRITICAL: No data was loaded from model or measurements. Skipping plot generation.")