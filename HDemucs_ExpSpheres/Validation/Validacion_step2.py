import os
import shutil

# ---------------------------
# Paths
# ---------------------------
validation_root = "/mnt/home/mdbm0009/DataBase_Demucs_Reorganizada/validation"
sdr_root = os.path.join(validation_root, "SDR")
os.makedirs(sdr_root, exist_ok=True)

for comb_folder in os.listdir(validation_root):
    comb_path = os.path.join(validation_root, comb_folder)
    if not os.path.isdir(comb_path) or comb_folder == "SDR":
        continue

    # Destination folder inside SDR
    comb_sdr_path = os.path.join(sdr_root, comb_folder)
    os.makedirs(comb_sdr_path, exist_ok=True)

    # ---------------------------
    # Walk through each combination
    # ---------------------------
    for root, dirs, files in os.walk(comb_path):
        # Copy all wavs from the folder that contains mixture.wav
        if "mixture.wav" in files:
            for file in files:
                if file.endswith(".wav"):
                    src = os.path.join(root, file)
                    dst = os.path.join(comb_sdr_path, file)
                    if not os.path.exists(dst):  # ✅ do not overwrite
                        shutil.copy2(src, dst)
                        print(f"Copied {file} -> {comb_sdr_path}")
                    else:
                        print(f"Skipped {file}, already exists")

        # Copy all wavs from demucs_output folder
        if os.path.basename(root) == "demucs_output":
            for file in files:
                if file.endswith(".wav"):
                    src = os.path.join(root, file)
                    dst = os.path.join(comb_sdr_path, file)
                    if not os.path.exists(dst):  # ✅ do not overwrite
                        shutil.copy2(src, dst)
                        print(f"Copied {file} -> {comb_sdr_path}")
                    else:
                        print(f"Skipped {file}, already exists")

        # Copy metadata.json if found
        if "metadata.json" in files:
            src = os.path.join(root, "metadata.json")
            dst = os.path.join(comb_sdr_path, "metadata.json")
            if not os.path.exists(dst):  # ✅ do not overwrite
                shutil.copy2(src, dst)
                print(f"Copied metadata.json -> {comb_sdr_path}")
            else:
                print("Skipped metadata.json, already exists")







