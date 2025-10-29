import os
import pathlib
import subprocess
import yaml
from pathlib import Path

# Root directory containing input subfolders with separated stems
RAIZ_CARPETAS = pathlib.Path('E:/DataBase_Demucs/inputs7')

   
# Paths to simulator configuration files
SIM_SETUP_DIR = Path("C:/Users/mabej/Desktop/TFM/code/TFM/SynthSOD_bleeding_sim/sim_setup")
CONFIG_FILE = SIM_SETUP_DIR / 'config.yaml'
DATASET_FILE = SIM_SETUP_DIR / 'dataset_files.json'
#CONFIG_FILE = "C:/Users/mabej/Desktop/TFM/code/TFM/SynthSOD_bleeding_sim/sim_setup/config.yaml"
#DATASET_FILE = "C:/Users/mabej/Desktop/TFM/code/TFM/SynthSOD_bleeding_sim/sim_setup/dataset_files.json"

# Simulation script
SIMULATE_SCRIPT = "C:/Users/mabej/Desktop/TFM/code/TFM/SynthSOD_bleeding_sim/simulate.py"

# Itera sobre cada subcarpeta
for subcarpeta in sorted(RAIZ_CARPETAS.iterdir()):
    if subcarpeta.is_dir():
        nombre = subcarpeta.name
        print(f"\nProcesando carpeta: {nombre}")

         # Custom input/output paths for this iteration
        nueva_fuente = str(subcarpeta.resolve())
        nueva_salida = str((Path('E:/DataBase_Demucs/outputs7') / nombre).resolve())

        # Load config.yaml and update paths
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)

        config['source_directory'] = nueva_fuente
        config['output_directory'] = nueva_salida

        # Save the updated config.yaml
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f)

        # Remove dataset_files.json if it exists
        if DATASET_FILE.exists():
            print("Eliminando dataset_files.json...")
            DATASET_FILE.unlink()

       
        # Launch the spatialization simulator
        subprocess.run(['python', str(SIMULATE_SCRIPT)], check=True)