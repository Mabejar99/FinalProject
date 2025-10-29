import os
import torch
import yaml
import soundfile as sf
from demucs.hdemucs import HDemucs
from demucs.apply import apply_model

# ---------------------------
# Configuration
# ---------------------------
checkpoint_path = "/mnt/home/mdbm0009/HDemucs_ExpSpheres/checkpoints/best_epoch=155.ckpt"
conf_path = "/mnt/home/mdbm0009/HDemucs_ExpSpheres/conf.yaml"
validation_root = "/mnt/home/mdbm0009/DataBase_Demucs_Reorganizada/validation"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")

print("Cargando modelo...")

# ---------------------------
# conf and model
# ---------------------------
with open(conf_path, "r") as f:
    conf = yaml.safe_load(f)

confdemucs = {
    "sources": [str(i) for i in range(conf["separator_conf"]["n_srcs"])],
    "audio_channels": conf["separator_conf"]["n_imics"],  
    "cac": conf["hdemucs_conf"]["cac"],
    "samplerate": conf["hdemucs_conf"]["samplerate"],
    "channels": conf["hdemucs_conf"]["channels"],
    "segment": conf["dataset"]["chunk_duration"],
}

model = HDemucs(**confdemucs).to(device)

# checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("auralossnew.")}
model.load_state_dict(state_dict)
model.eval()
print("Modelo cargado correctamente.")

# ---------------------------
# Output
# ---------------------------
def separate_and_save(mixture_path, save_folder):
    # Si la carpeta ya existe, no hacer nada
    if os.path.exists(save_folder):
        print(f"Ya existe {save_folder}, se omite este archivo.")
        return

    audio, sr = sf.read(mixture_path, dtype="float32", always_2d=True)
    audio = torch.tensor(audio.T, dtype=torch.float32)  # (C, T)

    #12 channels
    C, T = audio.shape
    if C < 12:
        pad = torch.zeros(12 - C, T, dtype=torch.float32)
        audio = torch.cat([audio, pad], dim=0)
    elif C > 12:
        audio = audio[:12, :]

    # (1, 12, T)
    audio = audio.unsqueeze(0).to(device)

    with torch.no_grad():
        estimates = apply_model(model, audio, shifts=0, device=device)[0]

    print("Estimates shape:", estimates.shape)

    os.makedirs(save_folder, exist_ok=True)

    # ---------------------------
    # Save
    # ---------------------------
    if estimates.dim() == 3:  
        all_out = estimates.permute(2, 1, 0).reshape(estimates.shape[2], -1).cpu().numpy()
        sf.write(os.path.join(save_folder, "Separated_All.wav"), all_out, sr)

        for i, src in enumerate(estimates):
            out = src.cpu().numpy().T  # (T, C)
            sf.write(os.path.join(save_folder, f"source_{i+1}.wav"), out, sr)

    elif estimates.dim() == 2:  
        all_out = estimates.cpu().numpy().T  
        sf.write(os.path.join(save_folder, "Separated_All.wav"), all_out, sr)

        for i, src in enumerate(estimates):
            out = src.cpu().numpy()  # (T,)
            sf.write(os.path.join(save_folder, f"source_{i+1}.wav"), out, sr)

# ---------------------------
# Folder
# ---------------------------
for subfolder in os.listdir(validation_root):
    subfolder_path = os.path.join(validation_root, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    # subfolder
    for random_folder in os.listdir(subfolder_path):
        random_path = os.path.join(subfolder_path, random_folder)
        if not os.path.isdir(random_path):
            continue

        # ---------------------------
        # mixture.wav
        # ---------------------------
        mixture_path = None

   
        candidate = os.path.join(random_path, "mixture.wav")
        if os.path.exists(candidate):
            mixture_path = candidate
        else:
            
            for sub in os.listdir(random_path):
                subfolder_candidate = os.path.join(random_path, sub)
                if os.path.isdir(subfolder_candidate) and sub.startswith("inputs"):
                    candidate = os.path.join(subfolder_candidate, "mixture.wav")
                    if os.path.exists(candidate):
                        mixture_path = candidate
                        break

        if mixture_path is not None:
            print(f"Procesando {mixture_path}")
            save_folder = os.path.join(random_path, "demucs_output")
            separate_and_save(mixture_path, save_folder)
        else:
            print(f"No encontrado mixture.wav en {random_path} ni en subcarpetas tipo 'inputs*'")






