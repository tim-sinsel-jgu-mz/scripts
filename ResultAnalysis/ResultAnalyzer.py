import netCDF4 as nc
import numpy as np

def read_air_temperature(filepath):
    """
    Reads the air temperature variable from an ENVI-met NetCDF output file.

    Args:
        filepath (str): The full path to the NetCDF file.
    """
    try:
        # Open the NetCDF file in read mode
        dataset = nc.Dataset(filepath, 'r')
        print("Successfully opened NetCDF file.")

        # --- Inspect the file (optional but recommended) ---
        print("\n--- File Metadata ---")
        print("Dimensions:", list(dataset.dimensions.keys()))
        print("Variables:", list(dataset.variables.keys()))
        
        # --- Identify the Air Temperature Variable ---
        # ENVI-met often uses 'T' or 'Ta' for air temperature.
        # You should check the variable list above to be certain.
        temp_variable_name = 'T' 

        if temp_variable_name in dataset.variables:
            print(f"\nFound air temperature variable: '{temp_variable_name}'")
            
            # Access the variable
            air_temp_var = dataset.variables[temp_variable_name]
            
            # Read the data into a numpy array
            # The [:] slice reads all data for that variable
            air_temp_data = air_temp_var[:]
            
            # --- Analyze the Data ---
            print(f"\n--- Data Analysis for '{temp_variable_name}' ---")
            print("Data type:", air_temp_data.dtype)
            
            # ENVI-met dimensions are typically (Time, Z, Y, X)
            print("Shape of data array (Time, Z, Y, X):", air_temp_data.shape)
            
            # Get variable attributes like units
            units = air_temp_var.units if 'units' in air_temp_var.ncattrs() else 'not specified'
            print("Units:", units)
            
            # --- Example: Access a specific data point ---
            # Let's get the temperature at the first time step, lowest height (z=0),
            # and a specific grid point (e.g., y=10, x=10)
            time_index = 0
            z_index = 0
            y_index = 10
            x_index = 10
            
            # Check if indices are within the bounds of the data shape
            if (time_index < air_temp_data.shape[0] and
                z_index < air_temp_data.shape[1] and
                y_index < air_temp_data.shape[2] and
                x_index < air_temp_data.shape[3]):
                
                specific_temp = air_temp_data[time_index, z_index, y_index, x_index]
                print(f"\nExample: Temperature at (t={time_index}, z={z_index}, y={y_index}, x={x_index}) is {specific_temp:.2f} {units}")
            else:
                print("\nExample indices (10,10) are out of bounds for this dataset's Y/X dimensions.")

        else:
            print(f"\nError: Variable '{temp_variable_name}' not found in the file.")
            print("Please check the list of available variables above and update the 'temp_variable_name' in the script.")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure the dataset is closed
        if 'dataset' in locals() and dataset.isopen():
            dataset.close()
            print("\nNetCDF file has been closed.")


if __name__ == '__main__':
    # --- IMPORTANT ---
    # Replace this with the actual path to your ENVI-met .nc file
    envi_met_file = 'your_envi_met_output.nc'
    
    read_air_temperature(envi_met_file)
