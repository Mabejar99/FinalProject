import sys
import types
import numpy as np
import sounddevice as sd
import torch
import yaml
from demucs.hdemucs import HDemucs
from demucs.apply import apply_model
import os
import soundfile as sf

# ----------------------
# Fix Demucs CPU / bitsandbytes
# ----------------------
torch_arange_orig = torch.arange
def arange_cpu(*args, **kwargs):
    kwargs.pop("device", None)
    return torch_arange_orig(*args, device="cpu", **kwargs)
torch.arange = arange_cpu

bitsandbytes = types.ModuleType('bitsandbytes')
optim = types.ModuleType('optim')
adam = types.ModuleType('adam')
class Adam8bit:
    def __init__(self, *args, **kwargs):
        pass
adam.Adam8bit = Adam8bit
optim.adam = adam
bitsandbytes.optim = optim
sys.modules['bitsandbytes'] = bitsandbytes
sys.modules['bitsandbytes.optim'] = optim
sys.modules['bitsandbytes.optim.adam'] = adam

# ----------------------
# Configuration
# ----------------------
fs = 48000
chunk_sec = 3
bh_total_channels = 64
demucs_channels = 12
bh_input_start = 0   # canales 1-12 de BlackHole (0-indexed)
bh_input_end = bh_input_start + demucs_channels
bh_output_start = 19 # enviar a canales 20-31 en Reaper
device_name = "BlackHole 64ch"
sd.default.device = device_name

silence_threshold = 1e-4
max_silent_blocks = 3

checkpoint_path = '/Users/mariabejar/Desktop/TFM/HDemucs_ExpSpheres/checkpoints/best_epoch=155.ckpt'
conf_path = './conf.yaml'
output_dir = "./demucs_output"
os.makedirs(output_dir, exist_ok=True)

# ----------------------
# Load Demucs model
# ----------------------
with open(conf_path,'r') as f:
    conf = yaml.safe_load(f)

confdemucs = {
    'sources': [str(i) for i in range(conf['separator_conf']['n_srcs'])],
    'audio_channels': conf['separator_conf']['n_imics'],
    'cac': conf['hdemucs_conf']['cac'],
    'samplerate': conf['hdemucs_conf']['samplerate'],
    'channels': conf['hdemucs_conf']['channels'],
    'segment': conf['dataset']['chunk_duration'],
}

model = HDemucs(**confdemucs).to('cpu')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("auralossnew.")}
model.load_state_dict(state_dict)
model.eval()
print("Modelo cargado.")

# ----------------------
# Function for processing blocks with Demucs
# ----------------------
def process_block(block_audio):
    tensor = torch.tensor(block_audio.T, dtype=torch.float32).unsqueeze(0)  # (1,C,T)
    with torch.no_grad():
        separated = apply_model(model, tensor, shifts=0, device='cpu')
    separated = separated / separated.abs().max()
    return separated[0].cpu().numpy().T  # (frames, channels)

# ----------------------
# Buffers 
# ----------------------
input_buffer = np.empty((0, demucs_channels), dtype=np.float32)
output_buffers = [np.empty((0,), dtype=np.float32) for _ in range(demucs_channels)]  # Para guardar al final
send_buffers = [np.empty((0,), dtype=np.float32) for _ in range(demucs_channels)]    # Para enviar en tiempo real
silent_counter = 0

# ----------------------
# Callback Stream
# ----------------------
def callback(indata, outdata, frames, time, status):
    global input_buffer, output_buffers, send_buffers, silent_counter

    if status:
        print(status)

    block = indata[:, bh_input_start:bh_input_end]
    input_buffer = np.vstack([input_buffer, block])

    # Silence detectiom
    rms = np.sqrt(np.mean(block**2, axis=0))
    if np.all(rms < silence_threshold):
        silent_counter += 1
    else:
        silent_counter = 0

    
    chunk_frames = chunk_sec * fs
    while input_buffer.shape[0] >= chunk_frames:
        process_chunk = input_buffer[:chunk_frames]
        input_buffer = input_buffer[chunk_frames:]
        separated_block = process_block(process_chunk)

        for c in range(demucs_channels):
            output_buffers[c] = np.hstack([output_buffers[c], separated_block[:, c]])  # Guardar todo
            send_buffers[c] = np.hstack([send_buffers[c], separated_block[:, c]])       # Para enviar a Reaper

    # Send to Reaper
    outdata[:] = np.zeros_like(outdata)
    send_frames = min(len(send_buffers[0]), frames)
    if send_frames > 0:
        send_block = np.zeros((frames, bh_total_channels), dtype=np.float32)
        for c in range(demucs_channels):
            send_block[:send_frames, bh_output_start + c] = send_buffers[c][:send_frames]
            send_buffers[c] = send_buffers[c][send_frames:]  # Solo recortamos send_buffers
        outdata[:] = send_block

    if silent_counter >= max_silent_blocks:
        raise sd.CallbackStop()

# ----------------------
# stream
# ----------------------
print(f"Recording, separating, and returning audio to Reaper by {device_name}...")
with sd.Stream(samplerate=fs, channels=bh_total_channels, dtype='float32',
               blocksize=fs*chunk_sec, callback=callback):
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Recording stopped by the user.")


# pending buffer

if input_buffer.shape[0] > 0:
    separated_block = process_block(input_buffer)
    for c in range(demucs_channels):
        output_buffers[c] = np.hstack([output_buffers[c], separated_block[:, c]])


# Save channel WAV. Testing

for c in range(demucs_channels):
    out_path = os.path.join(output_dir, f"source_channel_{c+1}.wav")
    sf.write(out_path, output_buffers[c], fs)
    print(f"Guardado: {out_path}")

