import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score

# --- PLOT STYLE ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 1.0 
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

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

            # LOGIK: ROOF vs FASSADE
            if loc_raw == 'ROOF':
                df['Location'] = loc_raw
                df['Distance'] = (df['z'] - roof_base_height).round(2)
            else:
                # FASSADEN: Nur z=1.75m und z=2.25m behalten
                mask_z = np.isclose(df['z'], 1.75, atol=0.05) | np.isclose(df['z'], 2.25, atol=0.05)
                df = df[mask_z]
                if df.empty: continue
                
                # Mittelwert über Höhen (eliminiert Duplikate und glättet Vertikal-Gradient)
                df = df.groupby('DateTime')['Temperature'].mean().reset_index()
                
                df['Location'] = loc_raw
                df['Distance'] = dist_filename
                df['z'] = 2.0 

            all_dfs.append(df[['DateTime', 'Location', 'Distance', 'z', 'Temperature']])
            
        except Exception:
            continue

    if not all_dfs: return pd.DataFrame()
    
    df_total = pd.concat(all_dfs, ignore_index=True)
    # Deduplizierung
    df_unique = df_total.groupby(['DateTime', 'Location', 'Distance', 'z'], as_index=False)['Temperature'].mean()
    
    try:
        csv_path = os.path.join(out_dir, f'DEBUG_ModelData_{label}.csv')
        df_unique.to_csv(csv_path, index=False)
    except: pass
    
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
    
    # Time Fix +3 min
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
    
    try:
        csv_path = os.path.join(out_dir, 'DEBUG_MeasData.csv')
        df_clean.to_csv(csv_path, index=False)
    except: pass
    
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

def calc_stats_safe(s_meas, s_model):
    try:
        s_meas_clean = s_meas.dropna()
        # Drop NaNs aus Modell (falls durch Glättung Ränder fehlen)
        s_model_clean = s_model.dropna()
        
        if s_meas_clean.empty or s_model_clean.empty: return np.nan, np.nan
        
        s_model_aligned = s_model_clean.reindex(s_meas_clean.index, method='nearest', tolerance=pd.Timedelta('30min'))
        mask = s_model_aligned.notna()
        y_true = s_meas_clean[mask]
        y_pred = s_model_aligned[mask]
        if len(y_true) < 5: return np.nan, np.nan
        return np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true, y_pred)
    except:
        return np.nan, np.nan

def run_plotting(df_meas, df_m1, df_m2, out_dir):
    print("\n--- Erstelle Plots ---")
    
    fig, axes = plt.subplots(7, 1, figsize=(10, 24), constrained_layout=True)
    
    pairs = [
        ('NORTH', 'NORTH'),
        ('ROOF', 'ROOF'),
        ('SOUTH1', 'SOUTH2'),
        ('SOUTH2', 'SOUTH1')
    ]
    dist_keys = [0.5, 1.5] 
    
    plot_idx = 0
    
    for meas_loc, mod_loc in pairs:
        for d_key in dist_keys:
            if meas_loc == 'ROOF' and d_key == 0.5: continue
            
            ax = axes[plot_idx]
            plot_idx += 1
            
            ts_meas = get_series_meas(df_meas, meas_loc, d_key)
            ts_m1 = get_series_model(df_m1, mod_loc, d_key)
            ts_m2 = get_series_model(df_m2, mod_loc, d_key)
            
            # --- GLÄTTUNG FÜR NEW (V59) ---
            # Rolling Mean: window=5 (bei 5min Daten = 25min), center=True
            if not ts_m2.empty:
                ts_m2 = ts_m2.rolling(window=5, center=True, min_periods=1).mean()
            
            # Debug Output
            label_dist = "05" if d_key == 0.5 else "15"
            debug_name = f"{meas_loc}_{label_dist}"
            
            if not ts_meas.empty:
                ts_meas.to_csv(os.path.join(out_dir, f"DEBUG_Plot_{debug_name}_Meas.csv"), header=["Temperature"])
                ax.plot(ts_meas.index, ts_meas, color='black', label='Meas', lw=1.0, alpha=0.8)
            
            stats_txt = []
            
            # V56
            if not ts_m1.empty:
                ts_m1.to_csv(os.path.join(out_dir, f"DEBUG_Plot_{debug_name}_V56.csv"), header=["Temperature"])
                ax.plot(ts_m1.index, ts_m1, color='#1f77b4', label='V56', lw=1, alpha=0.8)
                rmse, r2 = calc_stats_safe(ts_meas, ts_m1)
                if not np.isnan(rmse): 
                    stats_txt.append(f"V56: $R^2$={r2:.2f}, RMSE={rmse:.2f}")

            # New (Geglättet)
            if not ts_m2.empty:
                ts_m2.to_csv(os.path.join(out_dir, f"DEBUG_Plot_{debug_name}_NEW_Smoothed.csv"), header=["Temperature"])
                ax.plot(ts_m2.index, ts_m2, color='#d62728', label='V59', lw=1, alpha=0.8)
                rmse, r2 = calc_stats_safe(ts_meas, ts_m2)
                if not np.isnan(rmse): 
                    stats_txt.append(f"V59: $R^2$={r2:.2f}, RMSE={rmse:.2f}")
            
            if meas_loc == 'ROOF':
                title = "ROOF 1.0 m"
            else:
                lbl = "0.5 m" if d_key == 0.5 else "1.5 m"
                title = f"{meas_loc} {lbl}" 

            ax.set_title(title, fontweight='bold', loc='left', fontsize=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%Hh'))
            ax.grid(True, linestyle=':', alpha=0.6)
            
            ax.set_ylim(8, 27)
            ax.set_ylabel("Temp. (°C)")
            
            if stats_txt:
                t = "\n".join(stats_txt)
                ax.text(0.98, 0.02, t, transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))

    fpath = os.path.join(out_dir, 'Validation_Comparison.png')
    plt.savefig(fpath, dpi=300)
    fpath = os.path.join(out_dir, 'Validation_Comparison.svg')
    plt.savefig(fpath)
    print(f"Fertig! Gespeichert unter: {fpath}")

if __name__ == "__main__":
    dir_v1 = r'Y:\Danmark_Building\Danmark_Building_Validation_Long_V56\receptors'
    dir_v2 = r'Y:\Danmark_Building\Danmark_Building_Validation_Long_new\receptors'
    meas_file = r'D:\enviprojects\Projektwerkstatt_Nissen_Schoefl\Measurements_LongPeriod_1hCorr_ALLDISTS.csv'
    out_dir = r'D:\CompPlotsDanmark'
    
    ROOF_H = 20.0
    
    d_m1 = load_envimet_receptors(dir_v1, ROOF_H, "V56", out_dir)
    d_m2 = load_envimet_receptors(dir_v2, ROOF_H, "NEW", out_dir)
    d_meas = load_measurements(meas_file, out_dir)
    
    if not d_meas.empty:
        run_plotting(d_meas, d_m1, d_m2, out_dir)
    else:
        print("Abbruch: Keine Messdaten.")