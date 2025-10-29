import os
import numpy as np
import soundfile as sf
import pandas as pd
from safesdr import compute_track_sdr  

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
root_sdr = "/mnt/home/mdbm0009/DataBase_Demucs_Reorganizada/validation/SDR"
output_csv = os.path.join(root_sdr, "sdr_results.xlsx")
fs = 48000  # Sample rate del dataset

# ----------------------------
# FUNCIÓN PARA PROCESAR CARPETA
# ----------------------------
def process_combination_folder(folder):
    mixture_path = os.path.join(folder, "mixture.wav")
    refs_path = os.path.join(folder, "REFS.wav")

    if not (os.path.exists(mixture_path) and os.path.exists(refs_path)):
        print(f"No se encontró mixture.wav o REFS.wav en {folder}")
        return None

    # cargar mezcla y referencias
    mixture, _ = sf.read(mixture_path, dtype="float32", always_2d=True)
    mixture = mixture.T  # (C, T)
    refs, _ = sf.read(refs_path, dtype="float32", always_2d=True)
    refs = refs.T  # (S, T)

    # cargar todas las fuentes separadas
    sep_list = []
    active_channels = 0
    for i in range(refs.shape[0]):
        sep_path = os.path.join(folder, f"source_{i+1}.wav")
        if os.path.exists(sep_path):
            sep, _ = sf.read(sep_path, dtype="float32", always_2d=True)
            sep = sep.T  # (1, T) si es mono
        else:
            sep = np.zeros((1, refs.shape[1]), dtype="float32")
        sep_list.append(sep)
        if not np.allclose(refs[i], 0):
            active_channels += 1

    # concatenar por filas para que quede (S, T)
    separated = np.vstack(sep_list)  # (S, T)

    row = {"Combination": os.path.basename(folder)}

    # calcular SDR por canal
    sdr_sep = compute_track_sdr(separated, refs, fs)
    sdr_mix = compute_track_sdr(mixture, refs, fs)

    # calcular delta canal por canal
    sdr_delta = sdr_sep - sdr_mix

    # agregar resultados por canal
    for i, (val_sep, val_mix, val_delta) in enumerate(zip(sdr_sep, sdr_mix, sdr_delta)):
        row[f"SDR_Sep_Ch{i+1}"] = val_sep
        row[f"SDR_Mix_Ch{i+1}"] = val_mix
        row[f"SDR_Delta_Ch{i+1}"] = val_delta

    # estadísticas globales
    row["Num_Active_Channels"] = active_channels
    row["Mean_SDR_Mixture"] = np.nanmean(sdr_mix)
    row["Mean_SDR_Separated"] = np.nanmean(sdr_sep)
    row["Mean_Delta_SDR"] = np.nanmean(sdr_delta)

    # también podemos guardar la mediana global
    row["Median_SDR_Mixture"] = np.nanmedian(sdr_mix)
    row["Median_SDR_Separated"] = np.nanmedian(sdr_sep)
    row["Median_Delta_SDR"] = np.nanmedian(sdr_delta)

    return row

# ----------------------------
# RECORRER TODAS LAS CARPETAS
# ----------------------------
results = []
for comb in sorted(os.listdir(root_sdr)):
    comb_path = os.path.join(root_sdr, comb)
    if os.path.isdir(comb_path):
        print(f"Procesando {comb_path}")
        row = process_combination_folder(comb_path)
        if row:
            results.append(row)

# ----------------------------
# GUARDAR RESULTADOS EN EXCEL
# ----------------------------
df = pd.DataFrame(results)

# Round numeric columns a 3 decimales
numeric_cols = df.select_dtypes(include=[float, np.float32, np.float64]).columns
df[numeric_cols] = df[numeric_cols].round(3)

df.to_csv(output_csv, index=False)
print(f"\nResultados guardados en {output_csv}")



