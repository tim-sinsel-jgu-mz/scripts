import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import json
import numpy as np

# --------------------------
# Config
# --------------------------
# Globale Stileinstellungen
sns.set_theme(style="whitegrid", context="paper", font_scale=1)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.constrained_layout.use'] = True

# Datenkonfiguration
FILE_PATH = Path(r'D:\CompPlotsDanmark\FOX\Long_corr.fox')

# Zeitraum (7 Days)
START_DATETIME = "28.07.2018 04:00:00"
END_DATETIME = "04.08.2018 04:00:00"

# Plot Styling
NUM_Y_TICKS = 6
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 12
SHOW_GRID = True
GRID_STYLE = {'color': '#DDDDDD', 'linestyle': '--', 'linewidth': 0.2}

# Line Styles
STYLE_CONFIG = {
    'color': '#1f77b4',       # Standard Blue
    'linewidth': 1.5,
    'linestyle': '-',
}

VAR_STYLES = {
    'q': {'linestyle': '--', 'color': '#1f77b4'}, # Blue for Specific Humidity
    'wd': {'linestyle': '--', 'color': '#ff7f0e'}  # Orange for Wind Dir
}

# Labels
AXIS_LABELS = {
    'x': "Time",
    'y_sw_direct': "Direct Shortwave Radiation [W/m²]",
    'y_sw_diffuse': "Diffuse Shortwave Radiation [W/m²]",
    'y_sw': "Shortwave Radiation [W/m²]",
    'y_lw': "Longwave Radiation [W/m²]",
    'y_temp': "Air Temperature [°C]",
    'y_q': "Specific Humidity [g/kg]",
    'y_wind_speed': "Wind Speed [m/s]",
    'y_wind_direction': "Wind Direction [°]",
    'title_sw_direct': 'Direct Shortwave Radiation',
    'title_sw_diffuse': 'Diffuse Shortwave Radiation',
    'title_lw': 'Longwave Radiation',
    'title_temp_humidity': 'Air Temperature and Specific Humidity',
    'title_wind': 'Wind Conditions',
    'title_sw_direct_diffuse': 'Direct and Diffuse Shortwave Radiation'
}

# --------------------------
# Helper Functions
# --------------------------
def load_data(file_path):
    """Loads the FOX (JSON) file and extracts raw profiles."""
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return None

    print(f"Loading file: {file_path.name}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Access the list of timesteps
        timestep_list = data.get('timestepList', [])
        
        if not timestep_list:
            print("Error: 'timestepList' not found or empty in JSON.")
            return None

        # Process the list into a flat structure
        processed_data = []
        for item in timestep_list:
            record = {}
            
            # Basic info
            record['Date'] = item.get('date')
            record['Time'] = item.get('time')
            
            # Radiation
            record['directrad'] = item.get('swDir', 0)
            record['diffuserad'] = item.get('swDif', 0)
            record['lw'] = item.get('lwRad', 0)
            
            # Profiles - Extract value from first level (height 2m or 10m)
            # Temperature (tProfile)
            t_prof = item.get('tProfile', [])
            if t_prof:
                record['at'] = t_prof[0].get('value') # Kelvin
                
            # Specific Humidity (qProfile)
            q_prof = item.get('qProfile', [])
            if q_prof:
                record['q'] = q_prof[0].get('value') # g/kg
                
            # Wind (windProfile)
            w_prof = item.get('windProfile', [])
            if w_prof:
                record['ws'] = w_prof[0].get('wSpdValue')
                record['wd'] = w_prof[0].get('wDirValue')
            
            processed_data.append(record)

        # Create DataFrame
        df = pd.DataFrame(processed_data)
        
        # Create DateTime
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        # Clean up
        df = df.dropna(subset=['DateTime'])
        print(f"Successfully loaded {len(df)} rows.")
        return df

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def filter_data(df, start_datetime, end_datetime):
    if df is None or df.empty: return pd.DataFrame()
    return df[(df['DateTime'] >= start_datetime) & (df['DateTime'] <= end_datetime)]

def format_plot(ax, y_lim=None, num_yticks=None, y_label=None, x_lim=None, yticks=None):
    if y_lim: ax.set_ylim(y_lim)
    if x_lim: ax.set_xlim(x_lim)
        
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis='x', which='major', length=5, direction='in', labelsize=TICK_LABEL_SIZE)
    
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.tick_params(axis='y', which='major', length=3, direction='in', labelsize=TICK_LABEL_SIZE)
    elif num_yticks:
        ax.yaxis.set_major_locator(MaxNLocator(num_yticks))
        ax.tick_params(axis='y', which='major', length=3, direction='in', labelsize=TICK_LABEL_SIZE)
        
    ax.grid(SHOW_GRID, **GRID_STYLE)
    plt.setp(ax.get_xticklabels(), rotation=-45, ha='left')
    
    if y_label:
        ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_xlabel(AXIS_LABELS['x'], fontsize=AXIS_LABEL_SIZE, fontweight='bold')

# --------------------------
# Plotting Functions
# --------------------------

def plot_sw_radiation(df, start_dt, end_dt):
    fig, axdir = plt.subplots(figsize=(10, 6))

    if 'directrad' in df.columns:
        l1 = axdir.plot(df['DateTime'], df['directrad'], 
                      color=STYLE_CONFIG['color'], 
                      linewidth=STYLE_CONFIG['linewidth'], 
                      label='Direct')
    
    if 'diffuserad' in df.columns:
        l2 = axdir.plot(df['DateTime'], df['diffuserad'], 
                          color=VAR_STYLES['wd']['color'], 
                          linestyle=VAR_STYLES['wd']['linestyle'], 
                          linewidth=STYLE_CONFIG['linewidth'],
                          label='Diffuse')

    # Start at 0, break at 200 W/m² increments (5 ticks)
    sw_ylim = [0, 800]
    sw_ticks = np.arange(0, 801, 200)

    format_plot(axdir, y_lim=sw_ylim, yticks=sw_ticks, y_label=AXIS_LABELS['y_sw'], x_lim=[start_dt, end_dt])

    try:
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        axdir.legend(lns, labs, loc='upper right', frameon=False, prop={'size': LEGEND_FONT_SIZE})
    except:
        pass    

    plt.savefig(r"D:\CompPlotsDanmark\FOX\direct_and_diffuse_sw_radiation.svg", format='svg', bbox_inches='tight')
    plt.close()


def plot_temperature_humidity(df, start_dt, end_dt):
    fig, ax_temp = plt.subplots(figsize=(10, 6))
    ax_humidity = ax_temp.twinx()
    
    if 'at' in df.columns:
        l1 = ax_temp.plot(df['DateTime'], df['at'] - 273.15, 
                     color='#d62728', 
                     linewidth=STYLE_CONFIG['linewidth'], 
                     label='Air Temperature')
    
    if 'q' in df.columns:
        l2 = ax_humidity.plot(df['DateTime'], df['q'], 
                         color=VAR_STYLES['q']['color'], 
                         linestyle=VAR_STYLES['q']['linestyle'], 
                         linewidth=STYLE_CONFIG['linewidth'],
                         label='Specific Humidity')
        
    temp_ylim = [10, 25]
    humidity_ylim = [0, 20]
    
    temp_ticks = np.linspace(temp_ylim[0], temp_ylim[1], NUM_Y_TICKS)
    humidity_ticks = np.linspace(humidity_ylim[0], humidity_ylim[1], NUM_Y_TICKS)
    
    format_plot(ax_temp, y_lim=temp_ylim, yticks=temp_ticks, y_label=AXIS_LABELS['y_temp'], x_lim=[start_dt, end_dt])
    format_plot(ax_humidity, y_lim=humidity_ylim, yticks=humidity_ticks, y_label=AXIS_LABELS['y_q'])
    
    try:
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax_temp.legend(lns, labs, loc='upper left', frameon=False, prop={'size': LEGEND_FONT_SIZE})
    except:
        pass
        
    plt.savefig(r"D:\CompPlotsDanmark\FOX\temperature_humidity.svg", format='svg', bbox_inches='tight')
    plt.close()

def plot_wind(df, start_dt, end_dt):
    fig, ax_speed = plt.subplots(figsize=(10, 6))
    ax_direction = ax_speed.twinx()
    
    if 'ws' in df.columns:
        l1 = ax_speed.plot(df['DateTime'], df['ws'], 
                      color=STYLE_CONFIG['color'], 
                      linewidth=STYLE_CONFIG['linewidth'], 
                      label='Wind Speed')
    
    if 'wd' in df.columns:
        l2 = ax_direction.plot(df['DateTime'], df['wd'], 
                          color=VAR_STYLES['wd']['color'], 
                          linestyle=VAR_STYLES['wd']['linestyle'], 
                          linewidth=STYLE_CONFIG['linewidth'],
                          label='Wind Direction')

    # Changed max wind speed to 6 to generate 7 clean ticks (0, 1, 2, 3, 4, 5, 6)
    speed_ylim = [0, 3.6] 
    direction_ylim = [0, 360]
    
    speed_ticks = np.linspace(speed_ylim[0], speed_ylim[1], 7)
    direction_ticks = np.arange(0, 361, 60) # 7 Ticks: 0, 60, 120, 180, 240, 300, 360

    format_plot(ax_speed, y_lim=speed_ylim, yticks=speed_ticks, y_label=AXIS_LABELS['y_wind_speed'], x_lim=[start_dt, end_dt])
    format_plot(ax_direction, y_lim=direction_ylim, yticks=direction_ticks, y_label=AXIS_LABELS['y_wind_direction'])
    
    try:
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax_speed.legend(lns, labs, loc='lower right', frameon=False, prop={'size': LEGEND_FONT_SIZE})
    except:
        pass

    plt.savefig(r"D:\CompPlotsDanmark\FOX\wind.svg", format='svg', bbox_inches='tight')
    plt.close()

def main():
    df = load_data(FILE_PATH)
    
    if df is not None:
        start_dt = pd.to_datetime(START_DATETIME, format="%d.%m.%Y %H:%M:%S")
        end_dt = pd.to_datetime(END_DATETIME, format="%d.%m.%Y %H:%M:%S")
        df_filtered = filter_data(df, start_dt, end_dt)
        
        if df_filtered.empty:
            print(f"No data found in the time range {START_DATETIME} to {END_DATETIME}")
        else:
            plot_sw_radiation(df_filtered, start_dt, end_dt)
            plot_temperature_humidity(df_filtered, start_dt, end_dt)
            plot_wind(df_filtered, start_dt, end_dt)
            print("All plots created successfully.")

if __name__ == "__main__":
    main()