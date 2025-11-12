import os
import csv
import glob  # For finding files
import numpy as np
import pandas as pd  # For data loading and time-series alignment
import matplotlib.pyplot as plt  # For plotting
from sklearn.metrics import mean_squared_error, r2_score  # For stats

# --- Note on new libraries ---
# This script now requires pandas, matplotlib, and scikit-learn.
# You may need to install them using:
# pip install pandas matplotlib scikit-learn
# -----------------------------


def extract_receptor_data(file_paths, file_processing_map):
    """
    Extracts Potential Air Temperature and DateTime from ENVI-met receptor files.

    Args:
        file_paths (list): A list of full paths to the receptor files.
        file_processing_map (dict): Maps keywords (like 'ROOF') to target heights.

    Returns:
        dict: A dictionary where keys are short names (e.g., 'ROOF') and
              values are dicts containing 'datetimes' and 'temperatures'.
    """
    results = {}

    # Define the column names we need to find in the header
    z_col_name = 'z (m)'
    temp_col_name = 'Potential Air Temperature (°C)'
    dt_col_name = 'DateTime'

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        print(f"\n--- Processing file: {filename} ---")

        # Determine target height AND short name from the map
        target_height = None
        short_name = None
        for keyword, height in file_processing_map.items():
            if keyword in filename:
                target_height = height
                short_name = keyword
                break

        if target_height is None or short_name is None:
            print(f"Warning: File '{filename}' does not match any keywords in 'file_processing_map'. Skipping.")
            continue

        print(f"Target height: {target_height} m, Data name: '{short_name}'")

        datetimes = []
        temperatures = []
        try:
            # Use 'cp1252' (Windows 'ANSI') encoding as it was found to work
            with open(file_path, 'r', encoding='cp1252', errors='ignore') as f:
                header_line = f.readline().strip()
                header = [h.strip() for h in header_line.split(',')]

                try:
                    z_index = header.index(z_col_name)
                    temp_index = header.index(temp_col_name)
                    dt_index = header.index(dt_col_name)
                except ValueError:
                    print(f"Error: Could not find required columns in '{filename}'. Skipping.")
                    print("  Headers found:", header)
                    continue

                # Read the rest of the file for data
                for line in f:
                    if not line.strip():
                        continue

                    parts = [p.strip() for p in line.strip().split(',')]

                    if len(parts) <= max(z_index, temp_index, dt_index):
                        continue

                    try:
                        current_height = float(parts[z_index])

                        # Compare floating point numbers with a small tolerance
                        if abs(current_height - target_height) < 0.01:
                            potential_temp = float(parts[temp_index])
                            datetime_val = parts[dt_index]

                            temperatures.append(potential_temp)
                            datetimes.append(datetime_val)

                    except (ValueError, IndexError):
                        # Ignore lines where data is not a valid number
                        continue

            # Store both datetimes and temperatures
            results[short_name] = {
                'datetimes': datetimes,
                'temperatures': np.array(temperatures)
            }
            print(f"Successfully extracted {len(temperatures)} data points.")

        except FileNotFoundError:
            print(f"Error: File not found at '{file_path}'")
        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e}")

    return results


def write_data_to_csv(data_directory, extracted_data, output_filename):
    """
    Writes the extracted data to a single CSV file, using short names as headers.
    
    Returns:
        str: The full path to the saved CSV file, or None if failed.
    """
    if not extracted_data:
        print("No data was extracted, CSV file will not be created.")
        return None

    # Get the short names in the order they were processed
    output_names = list(extracted_data.keys())

    # Assume all files have the same datetimes, get them from the first file
    # This might fail if files have different timesteps, but it's the
    # simplest approach that works for this data.
    first_key = output_names[0]
    if not extracted_data[first_key]['datetimes']:
        print(f"Error: No datetimes found for '{first_key}'. Cannot write CSV.")
        return None
        
    all_datetimes = extracted_data[first_key]['datetimes']

    # Create the header row for the CSV
    csv_header = ['DateTime'] + output_names

    output_filepath = os.path.join(data_directory, output_filename)

    try:
        with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

            # Iterate through each timestep (row)
            for i in range(len(all_datetimes)):
                # Start the row with the datetime
                row = [all_datetimes[i]]

                # Add the temperature from each file for that datetime
                for name in output_names:
                    if i < len(extracted_data[name]['temperatures']):
                        row.append(extracted_data[name]['temperatures'][i])
                    else:
                        row.append('')  # Add empty string if data is missing

                # Write the completed row to the file
                writer.writerow(row)

        print(f"\nSuccessfully saved model data to '{output_filename}'")
        return output_filepath

    except IOError as e:
        print(f"\nError writing CSV file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")
        return None


def load_measurement_data(data_directory, measurement_filename):
    """
    Loads the measurement CSV file using pandas.
    Dynamically handles different column names found in "short" and "long" files.
    
    Returns:
        pandas.DataFrame: A DataFrame with a DateTime index and temperature columns,
                          or None if loading failed.
    """
    file_path = os.path.join(data_directory, measurement_filename)
    if not os.path.exists(file_path):
        print(f"Error: Measurement file not found at '{file_path}'. Cannot generate plots.")
        return None

    try:
        print(f"\n--- Loading measurement file: {measurement_filename} ---")

        # We use encoding='utf-8-sig' to automatically handle the BOM character
        # that causes the "column not found" error, even if the file is utf-8.
        df_meas = pd.read_csv(
            file_path,
            delimiter=',',
            decimal='.',
            encoding='utf-8' # Use utf-8-sig to handle potential BOM
        )

        # --- Robust Column Renaming ---
        # Strip any extra whitespace from column names
        df_meas.columns = df_meas.columns.str.strip()

        # Define all possible column names we're looking for
        # This now maps the two distinct "long" file columns
        possible_column_maps = {
            'Date and time': 'DateTime',
            
            # Set 1 (from Measurements_short.csv)
            'TAir South Fac 50 cm [°C]': 'SOUTH1_Meas', # Map short file to SOUTH1
            'TAir North Fac 50 cm [°C]': 'NORTH_Meas',
            'Tair Roof 100 cm [°C]': 'ROOF_Meas',
            
            # Set 2 (from Measurements.csv - long)
            'air temperature south facade 50 cm [C]': 'SOUTH1_Meas',
            'air temperature south facade close to ventilation 50 cm [C]': 'SOUTH2_Meas', # New mapping
            'air temperature north facade 50 cm [C]': 'NORTH_Meas',
            'air temperature roof 100 cm [C]': 'ROOF_Meas'
        }
        
        # Create a rename map *only* with the columns that actually exist in the file
        rename_map = {}
        for old_name, new_name in possible_column_maps.items():
            if old_name in df_meas.columns:
                rename_map[old_name] = new_name
                
        df_meas.rename(columns=rename_map, inplace=True)

        # --- Date Parsing ---
        if 'DateTime' not in df_meas.columns:
            print(f"Error: Could not find 'Date and time' column in {measurement_filename}.")
            print(f"  Detected columns: {list(df_meas.columns)}")
            return None
            
        try:
            # Now, parse the DateTime column using the correct format
            df_meas['DateTime'] = pd.to_datetime(df_meas['DateTime'], format='%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"Error parsing DateTime. Check format. Error: {e}")
            return None

        # Set the DateTime column as the index for easy alignment
        df_meas.set_index('DateTime', inplace=True)
        
        # Keep only the columns we need
        final_columns = ['SOUTH1_Meas', 'SOUTH2_Meas', 'NORTH_Meas', 'ROOF_Meas']
        
        # Filter down to columns that were actually found and renamed
        cols_to_keep = [col for col in final_columns if col in df_meas.columns]
        
        if not cols_to_keep:
            print("Error: No valid measurement columns (SOUTH1_Meas, etc.) were found.")
            return None
            
        df_meas = df_meas[cols_to_keep]
        
        # Drop rows where all measurement values are missing
        df_meas.dropna(how='all', inplace=True)
        
        print(f"Successfully loaded {len(df_meas)} measurement data points.")
        return df_meas

    except Exception as e:
        print(f"An error occurred loading measurement data: {e}")
        if "expected 1 fields" in str(e):
             print("  -> This often means the 'delimiter' is wrong.")
        if "Missing column" in str(e):
             print("  -> This often means the 'encoding' is wrong or the column name is misspelled.")
        return None


def generate_comparison_plots(model_csv_path, measurement_df, data_directory):
    """
    Generates comparison plots for model vs. measurement data.
    Dynamically compares SOUTH1/SOUTH2 to available measurement columns.
    """
    print("\n--- Generating Comparison Plots ---")
    try:
        # Load the model data we just saved
        df_model = pd.read_csv(model_csv_path)
        # Explicitly parse the model's datetime
        df_model['DateTime'] = pd.to_datetime(df_model['DateTime'], format='%Y-%m-%d %H:%M:%S')

        # --- FIX: Convert model data columns to numeric ---
        model_data_columns = [col for col in df_model.columns if col != 'DateTime']
        for col in model_data_columns:
            # errors='coerce' will turn any bad values (like '') into NaN
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

        df_model.set_index('DateTime', inplace=True)

        # --- This defines the *ideal* plot mapping ---
        plot_map = {
            'ROOF': 'ROOF_Meas',
            'NORTH': 'NORTH_Meas',
            'SOUTH1': 'SOUTH1_Meas', # Ideal: Compare SOUTH1 model to SOUTH1 sensor
            'SOUTH2': 'SOUTH2_Meas'  # Ideal: Compare SOUTH2 model to SOUTH2 sensor
        }

        # Create the output directory if it doesn't exist
        plot_dir = os.path.join(data_directory, 'comparison_plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            print(f"Created plot directory: {plot_dir}")

        # Now, create a plot for each item in our plot_map
        for model_col, meas_col_ideal in plot_map.items():
            
            meas_col = meas_col_ideal # This is the measurement column we'll actually use
            
            # --- DYNAMIC LOGIC for "Short" file ---
            # If we're trying to plot SOUTH2 vs SOUTH2_Meas...
            if model_col == 'SOUTH2' and meas_col_ideal not in measurement_df.columns:
                # ...and SOUTH2_Meas doesn't exist, check if SOUTH1_Meas *does* exist
                if 'SOUTH1_Meas' in measurement_df.columns:
                    print(f"Info: '{meas_col_ideal}' not found. Comparing '{model_col}' to 'SOUTH1_Meas' as a fallback.")
                    meas_col = 'SOUTH1_Meas' # Fallback to using the other sensor
                else:
                    # This handles if neither sensor is found
                    print(f"Warning: Neither '{meas_col_ideal}' nor fallback 'SOUTH1_Meas' found. Skipping plot for {model_col}.")
                    continue
            
            # --- Standard checks ---
            if model_col not in df_model.columns:
                print(f"Warning: Model data '{model_col}' not found. Skipping plot.")
                continue
            if meas_col not in measurement_df.columns:
                print(f"Warning: Measurement data '{meas_col}' not found. Skipping plot for {model_col}.")
                continue

            # --- 1. Get data for PLOTTING (raw data) ---
            y_true_series = measurement_df[meas_col].dropna()
            y_pred_series = df_model[model_col].dropna()

            if y_true_series.empty or y_pred_series.empty:
                print(f"Warning: No data for '{model_col}' or '{meas_col}'. Skipping plot.")
                continue

            # --- 2. Align data for STATS (using merge_asof) ---
            df_true_stats = y_true_series.to_frame().reset_index().sort_values(by='DateTime')
            df_pred_stats = y_pred_series.to_frame().reset_index().sort_values(by='DateTime')

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
                # --- 3. Calculate Statistics ---
                y_true = df_merged_stats[meas_col]
                y_pred = df_merged_stats[model_col]

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                info_text = f'R²: {r2:.3f}\nRMSE: {rmse:.3f} °C'
                print(f"Stats for {model_col} vs {meas_col}: R²={r2:.3f}, RMSE={rmse:.3f} °C")
            else:
                print(f"CRITICAL WARNING: No overlapping data found for '{model_col}' and '{meas_col}' within 10min tolerance.")

            # --- 4. Create Plot (using the raw, un-merged data) ---
            plt.figure(figsize=(15, 7))
            
            plot_title = f'Model vs. Measurement: {model_col}'
            meas_label = f'Measurement ({meas_col})'
            # Add a note to the title if we used the fallback
            if meas_col != meas_col_ideal: 
                plot_title = f'Model: {model_col}  vs.  Measurement: {meas_col}'
                meas_label = f'Measurement ({meas_col} - Fallback)'


            # Plot both with lines and markers
            plt.plot(y_true_series.index, y_true_series,
                     label=meas_label,
                     color='black', alpha=0.9, marker='.', markersize=4, linestyle='-')
            plt.plot(y_pred_series.index, y_pred_series,
                     label=f'Model ({model_col})',
                     color='red', alpha=0.8, marker='.', markersize=4, linestyle='-')

            plt.title(plot_title, fontsize=16)
            plt.xlabel('Date / Time', fontsize=12)
            plt.ylabel('Potential Air Temperature (°C)', fontsize=12)
            plt.legend()
            plt.grid(True, which='both', linestyle='--', alpha=0.6)

            # --- 5. Set X-axis to overlapping range ---
            start_time = max(y_true_series.index.min(), y_pred_series.index.min())
            end_time = min(y_true_series.index.max(), y_pred_series.index.max())

            if start_time < end_time:
                plt.xlim(start_time, end_time)

            # Add the stats box
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                     verticalalignment='top', fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            # Save the figure
            plot_filename = f'comparison_{model_col}_vs_{meas_col}.png'
            plot_filepath = os.path.join(plot_dir, plot_filename)
            plt.savefig(plot_filepath)
            plt.close()  # Close the figure to save memory

            print(f"Successfully created plot: {plot_filename}")

    except Exception as e:
        print(f"An error occurred during plot generation: {e}")


if __name__ == '__main__':
    # --- User-defined section ---
    # Example for "Short" run: r'd:/develop/projects/ENVImet_Results_Analyzer/Short'
    # Example for "Long" run:  r'd:/develop/projects/ENVImet_Results_Analyzer/Long'
    data_directory = r'd:/develop/projects/ENVImet_Results_Analyzer/Long'
    
    measurement_csv_name = 'Measurements.csv' 
        
    output_csv_name = 'extracted_model_data.csv'

    # This map defines *which* files to process and *what* height to use
    # NO TERRAIN
    file_processing_map_no_terrain = {
        'ROOF': 21.25,
        'NORTH': 1.75,
        'SOUTH1': 1.75,
        'SOUTH2': 1.75
    }
    # TERRAIN
    file_processing_map_terrain = {
        'ROOF': 28.50,
        'NORTH': 7.50,
        'SOUTH1': 8.50,
        'SOUTH2': 8.50
    }

    # --- End of User-defined section ---

    if not os.path.isdir(data_directory):
        print(f"CRITICAL ERROR: The specified directory does not exist:\n{data_directory}")
        print("Please update the 'data_directory' variable in the script.")
    else:
        # --- 1. Find all receptor files automatically ---
        search_path = os.path.join(data_directory, '*.AT_1DT')
        all_receptor_files = glob.glob(search_path)

        if 'Terrain' in data_directory:
            file_processing_map = file_processing_map_terrain
        else:
            file_processing_map = file_processing_map_no_terrain  

        print(file_processing_map)  

        if not all_receptor_files:
            print(f"CRITICAL ERROR: No '.AT_1DT' files found in '{data_directory}'.")
        else:
            # Filter files to only those matching our keywords
            receptor_files_to_process = []
            for f in all_receptor_files:
                for keyword in file_processing_map.keys():
                    if keyword in os.path.basename(f):
                        receptor_files_to_process.append(f)
                        break
            
            if not receptor_files_to_process:
                  print(f"CRITICAL ERROR: No files found matching keywords: {list(file_processing_map.keys())}")
                 
            else:
                print(f"Found {len(receptor_files_to_process)} matching receptor files to process.")

                # --- 2. Extract data from receptor files ---
                extracted_data = extract_receptor_data(receptor_files_to_process, file_processing_map)

                # --- 3. Write extracted data to a new CSV ---
                model_csv_path = write_data_to_csv(data_directory, extracted_data, output_csv_name)

                # --- 4. Load measurement data ---
                measurement_df = load_measurement_data(data_directory, measurement_csv_name)

                # --- 5. Generate plots (if both steps were successful) ---
                if model_csv_path and measurement_df is not None:
                    generate_comparison_plots(model_csv_path, measurement_df, data_directory)
                else:
                    print("\nSkipping plot generation due to errors in data loading.")

                # --- 6. Display final summary ---
                print("\n\n--- Extraction Summary: ---")
                if extracted_data:
                    for name, data in extracted_data.items():
                        print(f"\nData for '{name}':")
                        temp_array = data['temperatures']
                        if temp_array.size > 0:
                            print(f"  Timesteps: {len(temp_array)}, Avg Temp: {np.mean(temp_array):.2f}°C")
                        else:
                            print(f"  No data was found for the specified height level.")
                else:
                    print("No data was extracted. Check file content and height specifications.")

