import os
import pandas as pd

def process_folders(root_path):
    # Liste, um alle Daten für die Gesamt-CSV zu sammeln
    all_data_frames = []

    print(f"\nStarte Verarbeitung in: {root_path}\n")

    # Durchsuche den Ordner und alle Unterordner
    for subdir, dirs, files in os.walk(root_path):
        
        if subdir == root_path:
            continue

        folder_name = os.path.basename(subdir)
        current_folder_data = []

        files.sort() 
        found_files_count = 0
        
        for file in files:
            if file.startswith("26") and file.lower().endswith(".csv"):
                file_path = os.path.join(subdir, file)
                
                try:
                    # 1. Datei einlesen
                    df = pd.read_csv(file_path, header=None)
                    
                    # 2. Spalten verarbeiten
                    # Wir prüfen, ob genug Spalten da sind (mindestens 3 für Zeit, Temp, Feuchte)
                    if len(df.columns) >= 3:
                        # Wir benennen vorläufig alles, was wir erwarten (auch Wind, falls vorhanden)
                        # Damit pandas weiß, was was ist.
                        col_names = ['Timestamp', 'Temperature', 'Humidity', 'WindSpeed']
                        
                        # Falls die Datei weniger Spalten hat als unsere Liste, passen wir die Liste an
                        current_cols = col_names[:len(df.columns)]
                        df.columns = current_cols + list(df.columns[len(current_cols):])
                        
                        # --- ÄNDERUNG HIER ---
                        # Wir wählen explizit NUR diese 3 Spalten aus. 
                        # 'WindSpeed' wird hier ignoriert/weggeworfen.
                        df = df[['Timestamp', 'Temperature', 'Humidity']]
                    else:
                        print(f"  [Warnung] Datei {file} hat zu wenige Spalten.")
                        continue

                    # 3. Zeitstempel säubern
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                    df = df.dropna(subset=['Timestamp'])

                    # 4. Standort-Spalte hinzufügen
                    df['Standort'] = folder_name

                    current_folder_data.append(df)
                    found_files_count += 1
                    
                except Exception as e:
                    # Falls z.B. 'WindSpeed' gar nicht in der Datei war und der Zugriff fehlschlägt
                    print(f"  [Fehler] Datei {file}: {e}")

        # Wenn CSVs in diesem Ordner gefunden wurden:
        if current_folder_data:
            # --- Schritt A: CSV für den Subordner erstellen ---
            folder_df = pd.concat(current_folder_data, ignore_index=True)
            folder_df = folder_df.sort_values(by='Timestamp')

            output_filename = os.path.join(subdir, f"{folder_name}_komplett.csv")
            
            # Speichern ohne Wind-Spalte
            folder_df.to_csv(output_filename, index=False, sep=';', decimal=',')
            
            print(f"✔ {folder_name}: {found_files_count} Dateien -> {folder_name}_komplett.csv")

            all_data_frames.append(folder_df)

    # --- Schritt B: Gesamt-CSV erstellen ---
    if all_data_frames:
        print("\nErstelle Gesamtdatei aller Standorte...")
        master_df = pd.concat(all_data_frames, ignore_index=True)
        
        master_df = master_df.sort_values(by=['Standort', 'Timestamp'])
        
        master_output = os.path.join(root_path, "GESAMT_alle_Standorte.csv")
        master_df.to_csv(master_output, index=False, sep=';', decimal=',')
        
        print(f"FERTIG! Gesamtdatei gespeichert unter:\n-> {master_output}")
        print(f"Anzahl Datensätze gesamt: {len(master_df)}")
    else:
        print("\nKeine passenden Daten gefunden.")

if __name__ == "__main__":
    print("Start")
    user_path = r"Z:\\share\\M10\\M10_Messdaten_Coding"

    if os.path.isdir(user_path):
        process_folders(user_path)
    else:
        print(f"\nFehler: Der Pfad '{user_path}' konnte nicht gefunden werden.")