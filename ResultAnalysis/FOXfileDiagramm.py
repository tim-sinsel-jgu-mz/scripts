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
    'q': {'linestyle': '--', 'color': '#1f77b4'}, # Orange for Specific Humidity
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
    'y_q': "Specific Humidity [g/kg]", # Changed from Relative Humidity
    'y_wind_speed': "Wind Speed [m/s]",
    'y_wind_direction': "Wind Direction [°]",
    'title_sw_direct': 'Direct Shortwave Radiation',
    'title_sw_diffuse': 'Diffuse Shortwave Radiation',
    'title_lw': 'Longwave Radiation',
    'title_temp_humidity': 'Air Temperature and Specific Humidity', # Updated Title
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

def format_plot(ax, y_lim=None, num_yticks=None, y_label=None, x_lim=None):
    if y_lim: ax.set_ylim(y_lim)
    if x_lim: ax.set_xlim(x_lim)
        
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis='x', which='major', length=5, direction='in', labelsize=TICK_LABEL_SIZE)
    
    if num_yticks:
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
""" 
def plot_direct_sw_radiation(df, start_dt, end_dt):
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'directrad' in df.columns:
        ax.plot(df['DateTime'], df['directrad'], color=STYLE_CONFIG['color'], linewidth=STYLE_CONFIG['linewidth'], label='Direct SW')
    format_plot(ax, y_lim=[-3, 800], num_yticks=NUM_Y_TICKS, y_label=AXIS_LABELS['y_sw_direct'], x_lim=[start_dt, end_dt])
    ax.set_title(AXIS_LABELS['title_sw_direct'], fontsize=AXIS_LABEL_SIZE + 1, fontweight='bold', pad=20)
    plt.savefig(r"D:\CompPlotsDanmark\FOX\direct_sw_radiation.svg", format='svg', bbox_inches='tight')
    plt.close()

def plot_diffuse_sw_radiation(df, start_dt, end_dt):
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'diffuserad' in df.columns:
        ax.plot(df['DateTime'], df['diffuserad'], color=STYLE_CONFIG['color'], linewidth=STYLE_CONFIG['linewidth'], label='Diffuse SW')
    format_plot(ax, y_lim=[-2, 200], num_yticks=NUM_Y_TICKS, y_label=AXIS_LABELS['y_sw_diffuse'], x_lim=[start_dt, end_dt])
    ax.set_title(AXIS_LABELS['title_sw_diffuse'], fontsize=AXIS_LABEL_SIZE + 1, fontweight='bold', pad=20)
    plt.savefig(r"D:\CompPlotsDanmark\FOX\diffuse_sw_radiation.svg", format='svg', bbox_inches='tight')
    plt.close()

def plot_longwave_radiation(df, start_dt, end_dt):
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'lw' in df.columns:
        ax.plot(df['DateTime'], df['lw'], color=STYLE_CONFIG['color'], linewidth=STYLE_CONFIG['linewidth'], label='Longwave')
    format_plot(ax, y_lim=[350, 600], num_yticks=NUM_Y_TICKS, y_label=AXIS_LABELS['y_lw'], x_lim=[start_dt, end_dt])
    ax.set_title(AXIS_LABELS['title_lw'], fontsize=AXIS_LABEL_SIZE + 1, fontweight='bold', pad=20)
    plt.savefig(r"D:\CompPlotsDanmark\FOX\longwave_radiation.svg", format='svg', bbox_inches='tight')
    plt.close()
"""

def plot_sw_radiation(df, start_dt, end_dt):
    fig, axdir = plt.subplots(figsize=(10, 6))
    #ax_diff = axdir.twinx()

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

    format_plot(axdir, y_lim=[-3, 800], num_yticks=NUM_Y_TICKS, y_label=AXIS_LABELS['y_sw'], x_lim=[start_dt, end_dt])

    #axdir.set_title(AXIS_LABELS['title_sw_direct_diffuse'], fontsize=AXIS_LABEL_SIZE + 1, fontweight='bold', pad=20)

    # Legend
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
    
    # 1. Plot Temperature (Primary Axis)
    if 'at' in df.columns:
        # Convert Kelvin to Celsius
        l1 = ax_temp.plot(df['DateTime'], df['at'] - 273.15, 
                     color='#d62728', 
                     linewidth=STYLE_CONFIG['linewidth'], 
                     label='Air Temperature')
    
    # 2. Plot Specific Humidity (Secondary Axis)
    if 'q' in df.columns:
        l2 = ax_humidity.plot(df['DateTime'], df['q'], 
                         color=VAR_STYLES['q']['color'], 
                         linestyle=VAR_STYLES['q']['linestyle'], 
                         linewidth=STYLE_CONFIG['linewidth'],
                         label='Specific Humidity')
        
    format_plot(ax_temp, y_lim=[10, 25], num_yticks=NUM_Y_TICKS, y_label=AXIS_LABELS['y_temp'], x_lim=[start_dt, end_dt])
    
    # Specific Humidity Range (Adjusted for g/kg, typically 0-15 or 0-20 in summer)
    format_plot(ax_humidity, y_lim=[0, 20], num_yticks=NUM_Y_TICKS, y_label=AXIS_LABELS['y_q'])
    
    #ax_temp.set_title(AXIS_LABELS['title_temp_humidity'], fontsize=AXIS_LABEL_SIZE + 1, fontweight='bold', pad=20)
    
    # Legend
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

    format_plot(ax_speed, y_lim=[0, 4], num_yticks=NUM_Y_TICKS, y_label=AXIS_LABELS['y_wind_speed'], x_lim=[start_dt, end_dt])
    format_plot(ax_direction, y_lim=[0, 360], num_yticks=NUM_Y_TICKS, y_label=AXIS_LABELS['y_wind_direction'])
    #ax_speed.set_title(AXIS_LABELS['title_wind'], fontsize=AXIS_LABEL_SIZE + 1, fontweight='bold', pad=20)
    
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
            '''
            plot_direct_sw_radiation(df_filtered, start_dt, end_dt)
            plot_diffuse_sw_radiation(df_filtered, start_dt, end_dt)
            plot_longwave_radiation(df_filtered, start_dt, end_dt)
            '''
            plot_sw_radiation(df_filtered, start_dt, end_dt)
            plot_temperature_humidity(df_filtered, start_dt, end_dt)
            plot_wind(df_filtered, start_dt, end_dt)
            print("All plots created successfully.")

if __name__ == "__main__":
    main()