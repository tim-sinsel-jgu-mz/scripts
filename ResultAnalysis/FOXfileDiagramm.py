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
    'y_sw': "Shortwave Radiation [W/m²]",
    'y_temp': "Air Temperature [°C]",
    'y_q': "Specific Humidity [g/kg]",
    'y_wind_speed': "Wind Speed [m/s]",
    'y_wind_direction': "Wind Direction [°]",
    'title_sw_direct_diffuse': 'Direct and Diffuse Shortwave Radiation',
    'title_temp_humidity': 'Air Temperature and Specific Humidity',
    'title_wind': 'Wind Conditions'
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
        
        timestep_list = data.get('timestepList', [])
        
        if not timestep_list:
            print("Error: 'timestepList' not found or empty in JSON.")
            return None

        processed_data = []
        for item in timestep_list:
            record = {}
            record['Date'] = item.get('date')
            record['Time'] = item.get('time')
            record['directrad'] = item.get('swDir', 0)
            record['diffuserad'] = item.get('swDif', 0)
            record['lw'] = item.get('lwRad', 0)
            
            t_prof = item.get('tProfile', [])
            if t_prof:
                record['at'] = t_prof[0].get('value') # Kelvin
                
            q_prof = item.get('qProfile', [])
            if q_prof:
                record['q'] = q_prof[0].get('value') # g/kg
                
            w_prof = item.get('windProfile', [])
            if w_prof:
                record['ws'] = w_prof[0].get('wSpdValue')
                record['wd'] = w_prof[0].get('wDirValue')
            
            processed_data.append(record)

        df = pd.DataFrame(processed_data)
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df = df.dropna(subset=['DateTime'])
        print(f"Successfully loaded {len(df)} rows.")
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def filter_data(df, start_datetime, end_datetime):
    if df is None or df.empty: return pd.DataFrame()
    return df[(df['DateTime'] >= start_datetime) & (df['DateTime'] <= end_datetime)]

def format_plot(ax, y_lim=None, num_yticks=None, y_label=None, x_lim=None, yticks=None):
    if y_lim: ax.set_ylim(y_lim)
    if x_lim: ax.set_xlim(x_lim)
        
    # Updated Datetime formatting: "Day-Month"
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis='x', which='major', length=5, direction='in', labelsize=TICK_LABEL_SIZE)
    
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.tick_params(axis='y', which='major', length=3, direction='in', labelsize=TICK_LABEL_SIZE)
    elif num_yticks:
        ax.yaxis.set_major_locator(MaxNLocator(num_yticks))
        ax.tick_params(axis='y', which='major', length=3, direction='in', labelsize=TICK_LABEL_SIZE)
        
    ax.grid(SHOW_GRID, **GRID_STYLE)
    
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    
    if y_label:
        ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, fontweight='bold')

# --------------------------
# Plotting Functions
# --------------------------

def plot_sw_radiation(df, start_dt, end_dt, axdir):
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

    sw_ylim = [0, 800]
    sw_ticks = np.arange(0, 801, 200)

    format_plot(axdir, y_lim=sw_ylim, yticks=sw_ticks, y_label=AXIS_LABELS['y_sw'], x_lim=[start_dt, end_dt])
    #axdir.set_title(AXIS_LABELS['title_sw_direct_diffuse'], fontweight='bold', loc='left', fontsize=12)

    try:
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        axdir.legend(lns, labs, loc='upper right', frameon=False, prop={'size': LEGEND_FONT_SIZE})
    except:
        pass    


def plot_temperature_humidity(df, start_dt, end_dt, ax_temp):
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
    
    #ax_temp.set_title(AXIS_LABELS['title_temp_humidity'], fontweight='bold', loc='left', fontsize=12)
    
    try:
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax_temp.legend(lns, labs, loc='upper left', frameon=False, prop={'size': LEGEND_FONT_SIZE})
    except:
        pass


def plot_wind(df, start_dt, end_dt, ax_speed):
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

    speed_ylim = [0, 6] 
    direction_ylim = [0, 360]
    
    speed_ticks = np.linspace(speed_ylim[0], speed_ylim[1], 7)
    direction_ticks = np.arange(0, 361, 60)

    format_plot(ax_speed, y_lim=speed_ylim, yticks=speed_ticks, y_label=AXIS_LABELS['y_wind_speed'], x_lim=[start_dt, end_dt])
    format_plot(ax_direction, y_lim=direction_ylim, yticks=direction_ticks, y_label=AXIS_LABELS['y_wind_direction'])
    
    #ax_speed.set_title(AXIS_LABELS['title_wind'], fontweight='bold', loc='left', fontsize=12)
    
    try:
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax_speed.legend(lns, labs, loc='upper right', frameon=False, prop={'size': LEGEND_FONT_SIZE})
    except:
        pass


def main():
    df = load_data(FILE_PATH)
    
    if df is not None:
        start_dt = pd.to_datetime(START_DATETIME, format="%d.%m.%Y %H:%M:%S")
        end_dt = pd.to_datetime(END_DATETIME, format="%d.%m.%Y %H:%M:%S")
        df_filtered = filter_data(df, start_dt, end_dt)
        
        if df_filtered.empty:
            print(f"No data found in the time range {START_DATETIME} to {END_DATETIME}")
        else:
            # Create a single figure with 3 subplots stacked vertically
            fig, axes = plt.subplots(3, 1, figsize=(10, 10.5), constrained_layout=True)

            plot_temperature_humidity(df_filtered, start_dt, end_dt, axes[0])            
            plot_sw_radiation(df_filtered, start_dt, end_dt, axes[1])
            plot_wind(df_filtered, start_dt, end_dt, axes[2])
            
            # Save the combined plot
            out_path = r"D:\CompPlotsDanmark\FOX\combined_meteorology.svg"
            plt.savefig(out_path, format='svg', bbox_inches='tight')
            plt.close()
            print(f"Combined plot created successfully at {out_path}.")

if __name__ == "__main__":
    main()