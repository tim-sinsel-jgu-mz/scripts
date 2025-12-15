import os
import pandas as pd
import numpy as np

def process_data(meas_root, model_root):
    all_data_frames = []

    print(f"\n1. Messdaten-Ordner: {meas_root}")
    print(f"2. Modelldaten-Ordner: {model_root}\n")

    # Durchsuche den Messdaten-Ordner nach Unterordnern (Standorten)
    for subdir, dirs, files in os.walk(meas_root):
        
        if subdir == meas_root:
            continue

        folder_name = os.path.basename(subdir)
        print(f"Verarbeite Standort: {folder_name}...")
        
        # ---------------------------------------------------------
        # SCHRITT 1: Messdaten einlesen & Aufbereiten
        # ---------------------------------------------------------
        meas_files = [f for f in files if f.startswith("26") and f.lower().endswith(".csv")]
        meas_files.sort()
        
        current_meas_data = []
        
        for file in meas_files:
            file_path = os.path.join(subdir, file)
            try:
                # Einlesen ohne Header
                df = pd.read_csv(file_path, header=None)
                
                # Prüfen auf Mindestspaltenzahl
                if len(df.columns) >= 3:
                    # Temporäre Benennung zum Zugriff
                    # (Wir gehen davon aus: 0=Zeit, 1=Temp, 2=Feuchte)
                    col_mapping = {
                        0: 'Timestamp',
                        1: 'Temperature_measured',
                        2: 'Humidity_measured'
                    }
                    df = df.rename(columns=col_mapping)
                    
                    # Nur diese Spalten behalten
                    df = df[['Timestamp', 'Temperature_measured', 'Humidity_measured']]
                else:
                    continue

                # Zeitstempel parsen
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                df = df.dropna(subset=['Timestamp'])

                # --- NEU: Runden auf volle Minute ---
                df['Timestamp'] = df['Timestamp'].dt.round('min')

                current_meas_data.append(df)
            except Exception as e:
                print(f"  [Fehler Messdatei] {file}: {e}")

        if not current_meas_data:
            print("  -> Keine Messdaten gefunden.")
            continue

        # Alle Mess-Dateien dieses Standorts untereinander hängen
        df_meas = pd.concat(current_meas_data, ignore_index=True)
        
        # Falls durch das Runden mehrere Werte auf die gleiche Minute fallen (z.B. 12:00:05 und 12:00:55),
        # nehmen wir den Mittelwert, um Duplikate zu vermeiden.
        # Falls du lieber den ersten Wert willst, ersetze .mean() durch .first()
        df_meas = df_meas.groupby('Timestamp', as_index=False).mean()

        # ---------------------------------------------------------
        # SCHRITT 2: Modelldaten einlesen & Aufbereiten
        # ---------------------------------------------------------
        model_sub_path = os.path.join(model_root, folder_name)
        df_model_ready = None
        
        if os.path.isdir(model_sub_path):
            model_files = [f for f in os.listdir(model_sub_path) if f.endswith(".AT_1DT")]
            
            if model_files:
                m_file = model_files[0]
                m_path = os.path.join(model_sub_path, m_file)
                
                try:
                    # Einlesen (Latin-1 für ° Zeichen)
                    df_m = pd.read_csv(m_path, sep=',', skipinitialspace=True, encoding='latin-1')
                    df_m.columns = df_m.columns.str.strip() # Leerzeichen in Headern entfernen

                    # Prüfen ob Zeit und Höhe da sind
                    if 'z (m)' in df_m.columns and 'DateTime' in df_m.columns:
                        # Zeit parsen
                        df_m['DateTime'] = pd.to_datetime(df_m['DateTime'], errors='coerce')
                        df_m = df_m.dropna(subset=['DateTime'])
                        
                        # Filter: Höhe ca. 2.70 m
                        mask_height = np.isclose(df_m['z (m)'], 2.70, atol=0.05)
                        df_m_subset = df_m[mask_height].copy()
                        
                        if not df_m_subset.empty:
                            # Wir benennen die Modell-Zeit auch 'Timestamp' für den Merge
                            # Und mappen die Modell-Werte auf die gewünschten Namen
                            rename_map = {
                                'DateTime': 'Timestamp',
                                'Potential Air Temperature (°C)': 'Temperature_modelled',
                                'Relative Humidity (%)': 'Humidity_modelled'
                            }
                            
                            # Prüfen, ob die Quellspalten da sind
                            available_cols = [c for c in rename_map.keys() if c in df_m_subset.columns]
                            
                            if 'Potential Air Temperature (°C)' in available_cols and 'Relative Humidity (%)' in available_cols:
                                df_m_subset = df_m_subset[available_cols] # Nur diese behalten
                                df_m_subset = df_m_subset.rename(columns=rename_map)
                                
                                # Auch hier Zeitstempel sicherheitshalber runden (falls Modell Sekunden hat)
                                df_m_subset['Timestamp'] = df_m_subset['Timestamp'].dt.round('min')
                                
                                df_model_ready = df_m_subset
                                print(f"  -> Modelldaten OK: {len(df_model_ready)} Zeilen (z=2.7m) vorbereitet.")
                            else:
                                print(f"  [Warnung] Modelldatei fehlen Spalten. Vorhanden: {df_m_subset.columns.tolist()}")
                        else:
                            print(f"  [Warnung] Keine Daten für z=2.10m in {m_file}.")
                    else:
                        print(f"  [Warnung] Formatfehler in {m_file}.")
                except Exception as e:
                    print(f"  [Fehler Modelldatei] {m_file}: {e}")
            else:
                print("  -> Kein .AT_1DT File gefunden.")
        else:
            print(f"  -> Kein Modell-Ordner '{folder_name}'.")

        # ---------------------------------------------------------
        # SCHRITT 3: Merge (Inner Join)
        # ---------------------------------------------------------
        # Wir führen nur zusammen, wenn wir BEIDE Datensätze haben.
        if df_model_ready is not None and not df_meas.empty:
            
            # 'inner': Behält nur Zeitstempel, die in BEIDEN Dataframes vorkommen.
            combined_df = pd.merge(df_meas, df_model_ready, on='Timestamp', how='inner')
            
            if combined_df.empty:
                print("  [Info] Schnittmenge leer (Zeitstempel passen nicht zusammen).")
                continue
                
            # 'Site' Spalte hinzufügen (umbenannt von Standort)
            combined_df['Site'] = folder_name
            
            # ---------------------------------------------------------
            # SCHRITT 4: Spalten sortieren & Speichern
            # ---------------------------------------------------------
            # Gewünschte Reihenfolge:
            target_cols = ['Timestamp', 'Temperature_measured', 'Humidity_measured', 
                           'Temperature_modelled', 'Humidity_modelled', 'Site']
            
            # Sicherstellen, dass wir nur diese Spalten haben (falls beim Merge was übrig blieb)
            final_df = combined_df[target_cols]
            
            # Speichern im Subordner
            output_filename = os.path.join(subdir, f"{folder_name}_MERGED.csv")
            final_df.to_csv(output_filename, index=False, sep=';', decimal=',')
            
            all_data_frames.append(final_df)
            print(f"  -> ✔ Datei erstellt: {len(final_df)} Zeilen (Match von Messung & Modell).")

        else:
            print("  -> Übersprungen (Fehlende Mess- oder Modelldaten).")

    # ---------------------------------------------------------
    # SCHRITT 5: Gesamtdatei
    # ---------------------------------------------------------
    if all_data_frames:
        print("\nErstelle Gesamtdatei...")
        master_df = pd.concat(all_data_frames, ignore_index=True)
        
        # Sortieren nach Site, dann Zeit
        master_df = master_df.sort_values(by=['Site', 'Timestamp'])
        
        master_output = os.path.join(meas_root, "GESAMT_ALLE_STANDORTE_FINAL.csv")
        master_df.to_csv(master_output, index=False, sep=';', decimal=',')
        print(f"FERTIG! Datei gespeichert unter:\n{master_output}")
        print(f"Anzahl Zeilen gesamt: {len(master_df)}")
    else:
        print("\nKeine Daten erfolgreich zusammengeführt.")

if __name__ == "__main__":
    print("--- Daten-Merger: Messung + Simulation (Strict Match) ---")
    
    p1 = r"Z:\\share\\M10\\M10_Messdaten_Coding"
    p2 = r"Z:\\share\\M10\\M10_Modelldaten_Coding"
    
    if os.path.isdir(p1) and os.path.isdir(p2):
        process_data(p1, p2)
    else:
        print("\nFehler: Ungültiger Pfad.")