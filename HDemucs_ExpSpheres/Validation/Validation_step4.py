import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Update Excel
df = pd.read_excel("C:/Users/mabej/Desktop/TFM/code/TFM/SynthSOD_bleeding_sim/sdr_results.xlsx")

# Save figures
#output_dir = "C:/Users/mabej/Desktop/TFM/code/TFM/SynthSOD_bleeding_sim/figures"
#os.makedirs(output_dir, exist_ok=True)

# channels (1 a 12)
for n_channels in range(1, 13):
    df_n = df[df["Num_Active_Channels"] == n_channels]
    
    
    sdr_mix_cols = [f"SDR_Mix_Ch{i}" for i in range(1, n_channels+1) if f"SDR_Mix_Ch{i}" in df_n.columns]
    sdr_sep_cols = [f"SDR_Sep_Ch{i}" for i in range(1, n_channels+1) if f"SDR_Sep_Ch{i}" in df_n.columns]
    
    if len(sdr_mix_cols) == 0:
        continue  
    
    # NaN
    sdr_mix_data = df_n[sdr_mix_cols]
    sdr_sep_data = df_n[sdr_sep_cols]
    mask = (~sdr_mix_data.isna()).all(axis=1) & (~sdr_sep_data.isna()).all(axis=1)
    
    # /1000 because in the excel there is . instead of ,
    sdr_mix = sdr_mix_data[mask].values.flatten() / 1000
    sdr_sep = sdr_sep_data[mask].values.flatten() / 1000
    
    if len(sdr_mix) == 0:
        continue  
    
    # DeltaSDR 
    sdr_delta_calc = sdr_sep - sdr_mix

    # means
    mean_mix = np.mean(sdr_mix)
    mean_sep = np.mean(sdr_sep)
    mean_delta = np.mean(sdr_delta_calc)

    # boxplot
    data = [sdr_mix, sdr_sep, sdr_delta_calc]
    labels = ["SDR_Mixture", "SDR_Separated", "SDR_Delta"]

    
    plt.figure(figsize=(8,6))
    box = plt.boxplot(data, patch_artist=True, tick_labels=labels)
    colors = ["skyblue", "lightgreen", "lightcoral"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

  
    means = [mean_mix, mean_sep, mean_delta]
    for i, mean in enumerate(means, start=1):
        plt.plot(i, mean, "ro")
        plt.text(i, mean + 0.001, f"{mean:.3f}", ha="center", color="red")

    plt.title(f"Boxplot SDR ({n_channels} channel{'s' if n_channels>1 else ''})", fontsize=14)
    plt.ylabel("SDR (dB)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)


    # show figure
    plt.show()


