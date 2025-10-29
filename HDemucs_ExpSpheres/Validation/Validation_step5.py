import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar Excel
df = pd.read_excel("C:/Users/mabej/Desktop/TFM/code/TFM/SynthSOD_bleeding_sim/sdr_results.xlsx")

# Definir instrumentos y sus columnas correspondientes
instrument_map = {
    "Vocals": [1, 2],
    "Drums": [3, 4],
    "Bass": [5, 6],
    "Snare": [7],
    "Other": [8],
    "Crowd": [9],
    "Cymbal": [10],
    "AddInstr": [11],
    "Guitar": [12]
}

for inst, chs in instrument_map.items():
    sep_cols = [f"SDR_Sep_Ch{i}" for i in chs if f"SDR_Sep_Ch{i}" in df.columns]
    mix_cols = [f"SDR_Mix_Ch{i}" for i in chs if f"SDR_Mix_Ch{i}" in df.columns]

    if not sep_cols or not mix_cols:
        continue  # saltar si no existen las columnas

    # Calcular medias fila a fila si hay más de una columna
    sdr_sep = df[sep_cols].mean(axis=1) / 1000
    sdr_mix = df[mix_cols].mean(axis=1) / 1000
    sdr_delta = sdr_sep - sdr_mix

    # Eliminar NaN
    sdr_sep = sdr_sep.dropna()
    sdr_mix = sdr_mix.dropna()
    sdr_delta = sdr_delta.dropna()

    # Preparar datos para boxplot
    data = [sdr_mix.values, sdr_sep.values, sdr_delta.values]
    labels = ["SDR_Mixture", "SDR_Separated", "Delta"]

    plt.figure(figsize=(8, 6))
    box = plt.boxplot(data, patch_artist=True, tick_labels=labels)

    # Colorear cajas
    colors = ["skyblue", "lightgreen", "lightcoral"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    # Añadir medias
    means = [np.mean(arr) for arr in data]
    for i, mean in enumerate(means, start=1):
        plt.plot(i, mean, "ro")
        plt.text(i, mean + 0.01, f"{mean:.3f}", ha="center", color="red")

    plt.title(f"Boxplot SDR - {inst}", fontsize=14)
    plt.ylabel("SDR (dB)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

