import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# --- PLOT STYLE ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 1.0 
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.constrained_layout.use'] = True

def find_col(header, pattern):
    p = re.compile(pattern, re.IGNORECASE)
    for h in header:
        if p.search(h): return h
    return None

def load_envimet_receptors(data_directory: str, roof_base_height: float, label: str, out_dir: str) -> pd.DataFrame:
    print(f"--- [{label}] Lade Modell-Daten aus: {data_directory} ---")
    
    search_path = os.path.join(data_directory, '**', '*.AT_1DT')
    file_list = glob.glob(search_path, recursive=True)
    
    if not file_list:
        print(f"WARNUNG: Keine Dateien in {data_directory} gefunden.")
        return pd.DataFrame()

    all_dfs = []
    filename_pattern = re.compile(r'(NORTH|SOUTH1|SOUTH2|SOUTH|ROOF)(?:(?:_(\d{2}))|(?:(\d+)CM))?', re.IGNORECASE)

    for file_path in file_list:
        filename = os.path.basename(file_path)
        match = filename_pattern.search(filename)
        if not match: continue

        loc_raw = match.group(1).upper()
        if loc_raw == 'SOUTH': continue 
        
        dist_filename = None
        if match.group(2): dist_filename = float(match.group(2)) / 10.0
        elif match.group(3): dist_filename = float(match.group(3)) / 100.0
        
        try:
            with open(file_path, 'r', encoding='cp1252', errors='ignore') as f:
                header_line = f.readline()
                header = [h.strip() for h in header_line.split(',')]
            
            col_dt = find_col(header, r'DateTime')
            col_z = find_col(header, r'z \(m\)')
            col_temp = find_col(header, r'Potential Air Temperature')
            
            if not col_z or not col_temp or not col_dt: 
                continue

            df = pd.read_csv(file_path, skiprows=1, names=header, 
                             usecols=[col_dt, col_temp, col_z], 
                             encoding='cp1252', on_bad_lines='skip')
            
            df.rename(columns={col_temp: 'Temperature', col_z: 'z', col_dt: 'DateTime'}, inplace=True)
            
            df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            if df['DateTime'].isna().all():
                 df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True, errors='coerce')
            df['DateTime'] = df['DateTime'].dt.round('min')
            
            df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
            df['z'] = pd.to_numeric(df['z'], errors='coerce')
            df.dropna(inplace=True)

            # Sanity Check
            df = df[(df['Temperature'] > -40.0) & (df['Temperature'] < 60.0)]

            if loc_raw == 'ROOF':
                df['Location'] = loc_raw
                df['Distance'] = (df['z'] - roof_base_height).round(2)
            else:
                # FASSADEN: Nur z=1.75m und z=2.25m behalten
                mask_z = np.isclose(df['z'], 1.75, atol=0.05) | np.isclose(df['z'], 2.25, atol=0.05)
                df = df[mask_z]
                if df.empty: continue
                
                # Mittelwert über Höhen
                df = df.groupby('DateTime')['Temperature'].mean().reset_index()
                
                df['Location'] = loc_raw
                df['Distance'] = dist_filename
                df['z'] = 2.0 

            all_dfs.append(df[['DateTime', 'Location', 'Distance', 'z', 'Temperature']])
            
        except Exception:
            continue

    if not all_dfs: return pd.DataFrame()
    
    df_total = pd.concat(all_dfs, ignore_index=True)
    df_unique = df_total.groupby(['DateTime', 'Location', 'Distance', 'z'], as_index=False)['Temperature'].mean()
    
    return df_unique

def load_measurements(file_path: str, out_dir: str) -> pd.DataFrame:
    print(f"--- Lade Messdaten: {file_path} ---")
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
    except: return pd.DataFrame()
    
    dt_col = next((c for c in df.columns if 'Date' in c or 'Time' in c), None)
    if not dt_col: return pd.DataFrame()
    
    df.rename(columns={dt_col: 'DateTime'}, inplace=True)
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    df['DateTime'] = df['DateTime'] + pd.Timedelta(minutes=3)
    df['DateTime'] = df['DateTime'].dt.round('min')
    
    df.dropna(subset=['DateTime'], inplace=True)
    df.set_index('DateTime', inplace=True)

    df = df.stack().reset_index()
    df.columns = ['DateTime', 'Header', 'Temperature']
    
    pattern = re.compile(r'.*?(NORTH|SOUTH|ROOF)\s*(?:Fac|facade)?(.*?)\s*(\d+)\s*(cm|m)\s*\[.*', re.IGNORECASE)
    extracted = df['Header'].str.extract(pattern)
    
    if extracted.empty: return pd.DataFrame()
    extracted.columns = ['Loc', 'Vent', 'Dist', 'Unit']
    df = pd.concat([df, extracted], axis=1)
    
    def map_loc(row):
        l = str(row['Loc']).upper()
        v = str(row['Vent']).upper()
        if l == 'NORTH': return 'NORTH'
        if l == 'ROOF': return 'ROOF'
        if l == 'SOUTH': return 'SOUTH2' if 'VENTILATION' in v else 'SOUTH1'
        return None

    df['Location'] = df.apply(map_loc, axis=1)
    df['Dist'] = pd.to_numeric(df['Dist'], errors='coerce')
    df['Distance'] = np.where(df['Unit'].str.lower() == 'cm', df['Dist']/100, df['Dist'])
    df['Temperature'] = pd.to_numeric(df['Temperature'].astype(str).str.replace(',', '.'), errors='coerce')
    
    df_clean = df.dropna(subset=['Location', 'Distance', 'Temperature'])
    df_clean = df_clean.groupby(['DateTime', 'Location', 'Distance'], as_index=False)['Temperature'].mean()
    
    return df_clean

def get_series_model(df, location, dist_key):
    if df.empty: return pd.Series(dtype=float)

    df_loc = df[df['Location'] == location].copy()
    if df_loc.empty: return pd.Series(dtype=float)

    if location == 'ROOF':
        if dist_key == 1.5:
            target_z = 21.25
            df_loc['z_diff'] = abs(df_loc['z'] - target_z)
            valid = df_loc[df_loc['z_diff'] < 0.1]
            if valid.empty: return pd.Series(dtype=float)
            best_z = valid.loc[valid['z_diff'].idxmin()]['z']
            series = valid[valid['z'] == best_z].set_index('DateTime')['Temperature'].sort_index()
        else:
            return pd.Series(dtype=float)

    else: # FASSADEN
        df_loc['diff'] = abs(df_loc['Distance'] - dist_key)
        valid = df_loc[df_loc['diff'] < 0.15]
        if valid.empty: return pd.Series(dtype=float)
        
        best_dist = valid.loc[valid['diff'].idxmin()]['Distance']
        series = valid[valid['Distance'] == best_dist].set_index('DateTime')['Temperature'].sort_index()
    
    return series.groupby(level=0).mean()

def get_series_meas(df, location, dist_key):
    if df.empty: return pd.Series(dtype=float)

    df_loc = df[df['Location'] == location].copy()
    if df_loc.empty: return pd.Series(dtype=float)

    final_series = pd.Series(dtype=float)

    if location == 'ROOF':
        if dist_key == 1.5:
            target = 1.0
            df_loc['diff'] = abs(df_loc['Distance'] - target)
            valid = df_loc[df_loc['diff'] < 0.1]
            if not valid.empty:
                final_series = valid.set_index('DateTime')['Temperature'].sort_index()
        else:
            return pd.Series(dtype=float)

    elif dist_key == 1.5:
        s1 = pd.Series(dtype=float)
        s2 = pd.Series(dtype=float)
        
        diff1 = abs(df_loc['Distance'] - 1.0)
        valid1 = df_loc[diff1 < 0.1]
        if not valid1.empty:
            best1 = valid1.loc[abs(valid1['Distance'] - 1.0).idxmin()]['Distance']
            s1 = valid1[valid1['Distance'] == best1].set_index('DateTime')['Temperature']
            
        diff2 = abs(df_loc['Distance'] - 2.0)
        valid2 = df_loc[diff2 < 0.1]
        if not valid2.empty:
            best2 = valid2.loc[abs(valid2['Distance'] - 2.0).idxmin()]['Distance']
            s2 = valid2[valid2['Distance'] == best2].set_index('DateTime')['Temperature']
        
        if not s1.empty and not s2.empty:
             s1 = s1.groupby(level=0).mean()
             s2 = s2.groupby(level=0).mean()
             common = s1.index.intersection(s2.index)
             if not common.empty:
                 final_series = (s1.loc[common] + s2.loc[common]) / 2.0

    else: # 0.5m
        target = 0.5
        df_loc['diff'] = abs(df_loc['Distance'] - target)
        valid = df_loc[df_loc['diff'] < 0.15]
        
        if not valid.empty:
            best_dist = valid.loc[valid['diff'].idxmin()]['Distance']
            final_series = valid[valid['Distance'] == best_dist].set_index('DateTime')['Temperature'].sort_index()

    if not final_series.empty:
        final_series = final_series.groupby(level=0).mean()
        final_series = final_series.resample('5min').mean()
        
    return final_series

def get_aligned_data(s_meas, s_model):
    """Hilfsfunktion, um Mess- und Modelldaten zeitlich für Scatter-Plots abzugleichen."""
    try:
        s_meas_clean = s_meas.dropna()
        s_model_clean = s_model.dropna()
        if s_meas_clean.empty or s_model_clean.empty: 
            return None, None
        
        s_model_aligned = s_model_clean.reindex(s_meas_clean.index, method='nearest', tolerance=pd.Timedelta('30min'))
        mask = s_model_aligned.notna()
        return s_meas_clean[mask].values, s_model_aligned[mask].values
    except:
        return None, None

def calc_stats_safe(s_meas, s_model):
    x, y = get_aligned_data(s_meas, s_model)
    if x is None or len(x) < 5: 
        return np.nan, np.nan
    return np.sqrt(mean_squared_error(x, y)), r2_score(x, y)

def create_figure(plot_configs, df_meas, df_m1, df_m2, out_path, date_fmt='%d-%m', y_limits=(8, 27), title_suffix="", highlight_range=None):
    """
    Generiert eine Figure mit n Subplots (Zeitserien).
    Neu:
    - highlight_range: Tuple (start_dt, end_dt) für einen farbigen Hintergrundbereich (Schatten).
    """
    n_plots = len(plot_configs)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, n_plots * 3.5), constrained_layout=True)
    if n_plots == 1: axes = [axes]
    
    for i, config in enumerate(plot_configs):
        ax = axes[i]
        meas_loc = config['meas_loc']
        mod_loc = config['mod_loc']
        d_key = config['d_key']
        
        # --- NEU: Schatten zeichnen (im Hintergrund) ---
        if highlight_range is not None:
            # color='#FFFFE0' ist ein helles "LightYellow"
            ax.axvspan(highlight_range[0], highlight_range[1], color='#FFFFE0', alpha=0.6, lw=0)

        ts_meas = get_series_meas(df_meas, meas_loc, d_key)
        ts_m1 = get_series_model(df_m1, mod_loc, d_key)
        ts_m2 = get_series_model(df_m2, mod_loc, d_key)
        
        if not ts_m2.empty:
            ts_m2 = ts_m2.rolling(window=5, center=True, min_periods=1).mean()
        
        if not ts_meas.empty:
            ax.plot(ts_meas.index, ts_meas, color='black', label='Meas', lw=1.0, alpha=0.8)
        
        stats_txt = []
        
        if not ts_m1.empty:
            ax.plot(ts_m1.index, ts_m1, color='#1f77b4', label='V56', lw=1, alpha=0.8)
            rmse, r2 = calc_stats_safe(ts_meas, ts_m1)
            if not np.isnan(rmse): stats_txt.append(f"V56: R²={r2:.2f}, RMSE={rmse:.2f} K")

        if not ts_m2.empty:
            ax.plot(ts_m2.index, ts_m2, color='#d62728', label='V59', lw=1, alpha=0.8)
            rmse, r2 = calc_stats_safe(ts_meas, ts_m2)
            if not np.isnan(rmse): stats_txt.append(f"V59: R²={r2:.2f}, RMSE={rmse:.2f} K")
        
        # Titel & Achsen
        base_title = "ROOF 1.0 m" if meas_loc == 'ROOF' else f"{meas_loc} {'0.5 m' if d_key == 0.5 else '1.5 m'}" 
        full_title = f"{base_title} {title_suffix}"
        ax.set_title(full_title, fontweight='bold', loc='left', fontsize=12)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
        
        ax.grid(True, color='#DDDDDD', linestyle='--', linewidth=0.2)
        ax.tick_params(axis='both', which='major', colors='black', labelsize=12, direction='in', length=5)
        
        ax.set_ylim(y_limits)
        ax.set_ylabel("Temp. [°C]", fontsize=12, fontweight='bold')
        
        if stats_txt:
            t = "\n".join(stats_txt)
            ax.text(0.98, 0.05, t, transform=ax.transAxes, ha='right', va='bottom', fontsize=12,
                    bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#DDDDDD', linewidth=1))

    plt.savefig(out_path, dpi=300)
    print(f"Zeitreihe gespeichert unter: {out_path}")
    plt.close()

def create_regression_figure(plot_configs, df_meas, df_m1, df_m2, out_path):
    """Generiert eine Figure mit n Subplots untereinander (Regression Measured vs Modelled)."""
    n_plots = len(plot_configs)
    fig, axes = plt.subplots(n_plots, 1, figsize=(6, n_plots * 5), constrained_layout=True)
    if n_plots == 1: axes = [axes]
    
    for i, config in enumerate(plot_configs):
        ax = axes[i]
        meas_loc = config['meas_loc']
        mod_loc = config['mod_loc']
        d_key = config['d_key']
        
        ts_meas = get_series_meas(df_meas, meas_loc, d_key)
        ts_m1 = get_series_model(df_m1, mod_loc, d_key)
        ts_m2 = get_series_model(df_m2, mod_loc, d_key)
        
        # 1:1 Perfekte Regression (Diagonale)
        ax.plot([8, 27], [8, 27], color='black', linestyle='--', linewidth=1.5, label='1:1 Line')
        
        stats_txt = []
        
        # V56 Plotting & Regression
        if not ts_m1.empty:
            x, y = get_aligned_data(ts_meas, ts_m1)
            if x is not None and len(x) > 5:
                ax.scatter(x, y, alpha=0.4, color='#1f77b4', s=15, label='V56 Data')
                m, b = np.polyfit(x, y, 1)
                ax.plot(np.unique(x), np.poly1d((m, b))(np.unique(x)), color='#1f77b4', linewidth=2, label='V56 Fit')
                rmse, r2 = np.sqrt(mean_squared_error(x, y)), r2_score(x, y)
                stats_txt.append(f"V56: y = {m:.2f}x + {b:.2f} | R²={r2:.2f}")

        # V59/NEW Plotting & Regression
        if not ts_m2.empty:
            ts_m2 = ts_m2.rolling(window=5, center=True, min_periods=1).mean()
            x, y = get_aligned_data(ts_meas, ts_m2)
            if x is not None and len(x) > 5:
                ax.scatter(x, y, alpha=0.4, color='#d62728', s=15, label='V59 Data')
                m, b = np.polyfit(x, y, 1)
                ax.plot(np.unique(x), np.poly1d((m, b))(np.unique(x)), color='#d62728', linewidth=2, label='V59 Fit')
                rmse, r2 = np.sqrt(mean_squared_error(x, y)), r2_score(x, y)
                stats_txt.append(f"V59: y = {m:.2f}x + {b:.2f} | R²={r2:.2f}")
        
        title = "ROOF 1.0 m" if meas_loc == 'ROOF' else f"{meas_loc} {'0.5 m' if d_key == 0.5 else '1.5 m'}"
        ax.set_title(title, fontweight='bold', loc='left', fontsize=12)
        
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, color='#DDDDDD', linestyle='--', linewidth=0.2)
        ax.set_xlim(8, 27)
        ax.set_ylim(8, 27)
        ax.tick_params(axis='both', which='major', colors='black', labelsize=12, direction='in', length=7)
        ax.set_xlabel("Measured Temperature [°C]", fontsize=12, fontweight='bold')
        ax.set_ylabel("Modelled Temperature [°C]", fontsize=12, fontweight='bold')
        #ax.legend(loc='upper left', fontsize=10, frameon=True)
        
        if stats_txt:
            t = "\n".join(stats_txt)
            ax.text(0.98, 0.05, t, transform=ax.transAxes, ha='right', va='bottom', fontsize=12,
                    bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#DDDDDD', linewidth=1))

    plt.savefig(out_path, dpi=300)
    print(f"Regression gespeichert unter: {out_path}")
    plt.close()

def create_aggregated_regression_figure(all_configs, df_meas, df_m1, df_m2, out_path_base):
    """Erstellt einen aggregierten Regressionsplot über alle Distanzen & Sensoren hinweg."""
    
    # 1 Row, 2 Columns (V56 Links, V59 Rechts)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    
    all_x_m1, all_y_m1 = [], []
    all_x_m2, all_y_m2 = [], []
    
    # Iteriere über alle Konfigurationen um ein großes Set von Arrays aufzubauen
    for config in all_configs:
        meas_loc = config['meas_loc']
        mod_loc = config['mod_loc']
        d_key = config['d_key']
        
        ts_meas = get_series_meas(df_meas, meas_loc, d_key)
        ts_m1 = get_series_model(df_m1, mod_loc, d_key)
        ts_m2 = get_series_model(df_m2, mod_loc, d_key)
        
        if not ts_m2.empty:
            ts_m2 = ts_m2.rolling(window=5, center=True, min_periods=1).mean()
            
        if not ts_m1.empty:
            x1, y1 = get_aligned_data(ts_meas, ts_m1)
            if x1 is not None and len(x1) > 0:
                all_x_m1.extend(x1.tolist())
                all_y_m1.extend(y1.tolist())
                
        if not ts_m2.empty:
            x2, y2 = get_aligned_data(ts_meas, ts_m2)
            if x2 is not None and len(x2) > 0:
                all_x_m2.extend(x2.tolist())
                all_y_m2.extend(y2.tolist())

    # --- PLOT 1: V56 Aggregated ---
    ax1 = axes[0]
    ax1.plot([8, 27], [8, 27], color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    if all_x_m1:
        x, y = np.array(all_x_m1), np.array(all_y_m1)
        # alpha reduzierter aufgrund der großen Datenmenge
        ax1.scatter(x, y, alpha=0.25, color='#1f77b4', s=15, label='V56')
        m, b = np.polyfit(x, y, 1)
        ax1.plot(np.unique(x), np.poly1d((m, b))(np.unique(x)), color='black', linewidth=2, alpha=0.8, label='Lin. Regression V56')
        
        rmse, r2 = np.sqrt(mean_squared_error(x, y)), r2_score(x, y)
        ax1.text(0.98, 0.05, f"y = {m:.2f}x + {b:.2f}\nR²={r2:.2f}\nRMSE={rmse:.2f} K", 
                 transform=ax1.transAxes, ha='right', va='bottom', fontsize=12,
                 bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#DDDDDD', linewidth=1))

    ax1.set_title("V56", fontweight='bold', loc='left', fontsize=12)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, color='#DDDDDD', linestyle='--', linewidth=0.2)
    ax1.set_xlim(8, 27)
    ax1.set_ylim(8, 27)
    ax1.tick_params(axis='both', which='major', colors='black', labelsize=12, direction='in', length=7)
    ax1.set_xlabel("Measured Temperature [°C]", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Modelled Temperature [°C]", fontsize=12, fontweight='bold')
    #ax1.legend(loc='upper left', fontsize=10, frameon=True)

    # --- PLOT 2: V59 Aggregated ---
    ax2 = axes[1]
    ax2.plot([8, 27], [8, 27], color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    if all_x_m2:
        x, y = np.array(all_x_m2), np.array(all_y_m2)
        ax2.scatter(x, y, alpha=0.25, color='#d62728', s=15, label='V59')
        m, b = np.polyfit(x, y, 1)
        ax2.plot(np.unique(x), np.poly1d((m, b))(np.unique(x)), color='black', linewidth=2, alpha=0.8, label='Lin. Regression V59')
        
        rmse, r2 = np.sqrt(mean_squared_error(x, y)), r2_score(x, y)
        ax2.text(0.98, 0.05, f"y = {m:.2f}x + {b:.2f}\nR²={r2:.2f}\nRMSE={rmse:.2f} K", 
                 transform=ax2.transAxes, ha='right', va='bottom', fontsize=12,
                 bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#DDDDDD', linewidth=1))

    ax2.set_title("V59", fontweight='bold', loc='left', fontsize=12)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, color='#DDDDDD', linestyle='--', linewidth=0.2)
    ax2.set_xlim(8, 27)
    ax2.set_ylim(8, 27)
    ax2.tick_params(axis='both', which='major', colors='black', labelsize=12, direction='in', length=7)    
    ax2.set_xlabel("Measured Temperature [°C]", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Modelled Temperature [°C]", fontsize=12, fontweight='bold')
    #ax2.legend(loc='upper left', fontsize=10, frameon=True)

    # Speichern der aggregierten Plots in PNG und SVG
    plt.savefig(f"{out_path_base}.png", dpi=300)
    plt.savefig(f"{out_path_base}.svg")
    print(f"Aggregierte Regression gespeichert unter: {out_path_base}.png und {out_path_base}.svg")
    plt.close()

def run_plotting_split(df_meas, df_m1, df_m2, out_dir):
    print("\n--- Erstelle getrennte Plots ---")
    
    # 1. Konfiguration für ROOF & 1.5m
    config_fig1 = [
        {'meas_loc': 'ROOF',   'mod_loc': 'ROOF',   'd_key': 1.5},
        {'meas_loc': 'NORTH',  'mod_loc': 'NORTH',  'd_key': 1.5},
        {'meas_loc': 'SOUTH1', 'mod_loc': 'SOUTH2', 'd_key': 1.5},
        {'meas_loc': 'SOUTH2', 'mod_loc': 'SOUTH1', 'd_key': 1.5}
    ]
    # Zeitreihen
    create_figure(config_fig1, df_meas, df_m1, df_m2, os.path.join(out_dir, 'Validation_High_Level_TS.png'))
    create_figure(config_fig1, df_meas, df_m1, df_m2, os.path.join(out_dir, 'Validation_High_Level_TS.svg'))
    # Einzelne Regressionen
    create_regression_figure(config_fig1, df_meas, df_m1, df_m2, os.path.join(out_dir, 'Validation_High_Level_Reg.png'))
    create_regression_figure(config_fig1, df_meas, df_m1, df_m2, os.path.join(out_dir, 'Validation_High_Level_Reg.svg'))
    
    # 2. Konfiguration für 0.5m
    config_fig2 = [
        {'meas_loc': 'NORTH',  'mod_loc': 'NORTH',  'd_key': 0.5},
        {'meas_loc': 'SOUTH1', 'mod_loc': 'SOUTH2', 'd_key': 0.5},
        {'meas_loc': 'SOUTH2', 'mod_loc': 'SOUTH1', 'd_key': 0.5}
    ]
    # Zeitreihen
    create_figure(config_fig2, df_meas, df_m1, df_m2, os.path.join(out_dir, 'Validation_Low_Level_TS.png'))
    create_figure(config_fig2, df_meas, df_m1, df_m2, os.path.join(out_dir, 'Validation_Low_Level_TS.svg'))
    # Einzelne Regressionen
    create_regression_figure(config_fig2, df_meas, df_m1, df_m2, os.path.join(out_dir, 'Validation_Low_Level_Reg.png'))
    create_regression_figure(config_fig2, df_meas, df_m1, df_m2, os.path.join(out_dir, 'Validation_Low_Level_Reg.svg'))

    # 3. Aggregierte Regression über alle Konfigurationen
    all_configs = config_fig1 + config_fig2
    agg_out_base = os.path.join(out_dir, 'Validation_Aggregated_Reg')
    create_aggregated_regression_figure(all_configs, df_meas, df_m1, df_m2, agg_out_base)

def create_profile_figure(df_meas, df_m1, df_m2, target_time, out_path):
    """
    Erstellt ein 2x2 Profil-Plot für einen Zeitpunkt. 
    X-Achse geht nun bis 2.2m für alle Plots.
    """
    t_ts = pd.to_datetime(target_time)
    
    def get_profile(df, loc):
        if df.empty: return pd.DataFrame()
        # Zeitfilter
        mask_time = df['DateTime'] == t_ts
        d = df[mask_time & (df['Location'] == loc)].copy()
        if d.empty: return pd.DataFrame()
        
        # Filter: Nur Punkte bis 2.5m (um sicher zu sein, dass 2.2m draufpasst)
        d = d[d['Distance'] <= 2.5]
        return d.sort_values('Distance')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    sites = [
        ('NORTH', axes[0, 0]), 
        ('SOUTH1', axes[0, 1]), 
        ('SOUTH2', axes[1, 0]), 
        ('ROOF', axes[1, 1])
    ]
    
    fig.suptitle(f"Temperature Profile @ {t_ts.strftime('%d.%m.%Y %H:%M')}", fontsize=14, fontweight='bold')

    for loc, ax in sites:
        p_meas = get_profile(df_meas, loc)
        p_m1 = get_profile(df_m1, loc)
        p_m2 = get_profile(df_m2, loc)
        
        # Plotting
        if not p_meas.empty:
            ax.plot(p_meas['Distance'], p_meas['Temperature'], 'o-', color='black', label='Meas', markersize=6, lw=1.5)
        if not p_m1.empty:
            ax.plot(p_m1['Distance'], p_m1['Temperature'], 's-', color='#1f77b4', label='V56', markersize=5, alpha=0.8)
        if not p_m2.empty:
            ax.plot(p_m2['Distance'], p_m2['Temperature'], '^-', color='#d62728', label='V59', markersize=5, alpha=0.8)

        # Styling
        ax.set_title(loc, fontweight='bold', fontsize=12)
        ax.grid(True, color='#DDDDDD', linestyle='--', linewidth=0.5)
        
        xlabel = "Height above Roof [m]" if loc == 'ROOF' else "Distance to Wall [m]"
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Temperature [°C]", fontsize=10)
        
        # NEU: Limit fix auf 0 bis 2.2m
        ax.set_xlim(0, 2.2)
        
        if loc == 'NORTH':
            ax.legend(fontsize=10, frameon=True, facecolor='white', framealpha=1)

    plt.savefig(out_path)
    plt.savefig(out_path.replace('.png', '.svg'))
    print(f"Profil-Plot gespeichert: {out_path}")
    plt.close()

def create_contour_figure(df_meas, df_m1, df_m2, location, out_path, mode='all'):
    """
    Erstellt Contour-Plots mit vertikaler Glättung und Tag/Nacht-Filter.
    mode: 'all', 'day' (06:00-21:00), 'night' (21:00-06:00)
    """
    
    def prep_grid(df, loc):
        if df.empty: return None, None, None, None, None
        
        d = df[df['Location'] == loc].copy()
        
        # --- TIME FILTER ---
        if mode == 'day':
            d = d[(d['DateTime'].dt.hour >= 6) & (d['DateTime'].dt.hour < 21)]
        elif mode == 'night':
            d = d[(d['DateTime'].dt.hour >= 21) | (d['DateTime'].dt.hour < 6)]
            
        d = d[d['Distance'] <= 2.5] 
        if d.empty: return None, None, None, None, None
        
        # Pivot & Temporal Interp
        piv = d.pivot_table(index='DateTime', columns='Distance', values='Temperature')
        
        # Resample fügt Lücken für die ausgeblendeten Zeiten ein (z.B. Nacht bei mode='day')
        # Das sorgt für saubere Trennung der Tage im Plot (weiße Streifen)
        piv = piv.resample('10min').interpolate(method='time', limit=6)
        
        # CRASH FIX: Mindestens 2 Höhen benötigt
        if piv.shape[1] < 2: return None, None, None, None, None
        if piv.empty or piv.isna().all().all(): return None, None, None, None, None
        
        # Spatial Smoothing (Vertikale Interpolation)
        try:
            df_trans = piv.T 
            fine_index = np.arange(df_trans.index.min(), df_trans.index.max() + 0.05, 0.05)
            combined_index = df_trans.index.union(fine_index).sort_values()
            df_trans = df_trans.reindex(combined_index).interpolate(method='index', limit_direction='both')
            piv = df_trans.reindex(fine_index).T
        except: pass

        X, Y = np.meshgrid(mdates.date2num(piv.index), piv.columns)
        return X, Y, piv.values.T, np.nanmin(piv.values), np.nanmax(piv.values)

    X_m, Y_m, Z_m, min_m, max_m = prep_grid(df_meas, location)
    X_1, Y_1, Z_1, min_1, max_1 = prep_grid(df_m1, location)
    X_2, Y_2, Z_2, min_2, max_2 = prep_grid(df_m2, location)

    mins = [m for m in [min_m, min_1, min_2] if m is not None and not np.isnan(m)]
    maxs = [m for m in [max_m, max_1, max_2] if m is not None and not np.isnan(m)]
    
    if not mins: 
        print(f"Keine validen Daten für {location} ({mode}), Plot wird übersprungen.")
        return # Skip empty plots

    vmin, vmax = min(mins), max(maxs)
    if vmin == vmax: vmin -= 1; vmax += 1

    levels = np.linspace(np.floor(vmin), np.ceil(vmax), 40)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True, sharex=True)
    
    def do_plot(ax, X, Y, Z, title):
        ax.set_ylabel("Distance [m]", fontweight='bold')
        ax.set_title(title, fontweight='bold', loc='left')
        
        if X is not None:
            cf = ax.contourf(X, Y, Z, levels=levels, cmap='RdYlBu_r', extend='both')
            ax.grid(True, color='white', linestyle=':', alpha=0.3)
            return cf
        else:
            # Fallback
            ax.text(0.5, 0.5, "Not enough vertical data\n(Needs >1 height level)", 
                    ha='center', va='center', transform=ax.transAxes, color='gray')
            ax.set_yticks([])
            return None

    # Plotting
    cf1 = do_plot(axes[0], X_m, Y_m, Z_m, f"Measured: {location} ({mode.upper()})")
    do_plot(axes[1], X_1, Y_1, Z_1, f"Model V56: {location} ({mode.upper()})")
    cf_last = do_plot(axes[2], X_2, Y_2, Z_2, f"Model V59: {location} ({mode.upper()})")

    # X-Achse
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    axes[2].set_xlabel("Date / Time", fontweight='bold')
    
    # Colorbar
    mappable = cf1 if cf1 else (cf_last if cf_last else None)
    if mappable:
        cbar = fig.colorbar(mappable, ax=axes, location='right', aspect=40, pad=0.02)
        cbar.set_label("Temperature [°C]", fontsize=12)

    plt.savefig(out_path)
    print(f"Contour-Plot gespeichert: {out_path}")
    plt.close()

def run_single_day_plot(df_meas, df_m1, df_m2, out_dir):
    print("\n--- Erstelle Single Day Plot (NORTH, 31.07.) ---")
    
    start_dt = pd.to_datetime("2022-07-31 00:00:00")
    end_dt = pd.to_datetime("2022-08-01 00:00:00")
    
    # NEU: Definition des Zeitfensters für den gelben Schatten
    shadow_start = pd.to_datetime("2022-07-31 05:00:00")
    shadow_end   = pd.to_datetime("2022-07-31 06:00:00")
    
    def filter_time(df):
        if df.empty: return df
        mask = (df['DateTime'] >= start_dt) & (df['DateTime'] <= end_dt)
        return df.loc[mask].copy()

    d_meas_sub = filter_time(df_meas)
    d_m1_sub = filter_time(df_m1)
    d_m2_sub = filter_time(df_m2)
    
    if d_meas_sub.empty:
        print("Warnung: Keine Daten für den gewählten Zeitraum gefunden.")
        return

    plot_config = [
        {'meas_loc': 'NORTH', 'mod_loc': 'NORTH', 'd_key': 0.5},
        {'meas_loc': 'NORTH', 'mod_loc': 'NORTH', 'd_key': 1.5}
    ]
    
    out_filename = "Validation_NORTH_SingleDay_3107.png"
    out_path = os.path.join(out_dir, out_filename)
    
    # Aufruf mit highlight_range Parameter
    create_figure(
        plot_config, 
        d_meas_sub, 
        d_m1_sub, 
        d_m2_sub, 
        out_path, 
        date_fmt='%H:%M', 
        y_limits=(12, 24),
        title_suffix="(31-07-2022)",
        highlight_range=(shadow_start, shadow_end)
    )
    
    create_figure(
        plot_config, 
        d_meas_sub, 
        d_m1_sub, 
        d_m2_sub, 
        out_path.replace('.png', '.svg'), 
        date_fmt='%H:%M', 
        y_limits=(12, 24),
        title_suffix="(31-07-2022)",
        highlight_range=(shadow_start, shadow_end)
    )

if __name__ == "__main__":
    dir_v1 = r'Y:\Danmark_Building\Danmark_Building_Validation_Long_V56\receptors'
    dir_v2 = r'Y:\Danmark_Building\Danmark_Building_Validation_Long_new\receptors'
    meas_file = r'D:\enviprojects\Projektwerkstatt_Nissen_Schoefl\Measurements_LongPeriod_1hCorr_ALLDISTS.csv'
    out_dir = r'D:\CompPlotsDanmark'
    
    ROOF_H = 20.0
    
    # 1. Daten laden
    d_m1 = load_envimet_receptors(dir_v1, ROOF_H, "V56", out_dir)
    d_m2 = load_envimet_receptors(dir_v2, ROOF_H, "NEW", out_dir)
    d_meas = load_measurements(meas_file, out_dir)
    
    if not d_meas.empty:
        # Bestehende Plots ausführen (optional, falls gewünscht)
        run_plotting_split(d_meas, d_m1, d_m2, out_dir)

        modes = ['day', 'night']
        locations = ['NORTH', 'SOUTH1', 'SOUTH2', 'ROOF']
        
        for mode in modes:
            print(f"\n--- Generiere Contour Plots ({mode.upper()}) ---")
            for loc in locations:
                filename = f'Contour_{loc}_{mode}.png'
                create_contour_figure(d_meas, d_m1, d_m2, loc, os.path.join(out_dir, filename), mode=mode)

        # Profil-Plot für 30.07.2022 um 13:00 Uhr
        target_time = "2022-07-30 13:00:00"
        create_profile_figure(d_meas, d_m1, d_m2, target_time, os.path.join(out_dir, 'Validation_Profile_1300.png'))

        run_single_day_plot(d_meas, d_m1, d_m2, out_dir)
    else:
        print("Abbruch: Keine Messdaten.")