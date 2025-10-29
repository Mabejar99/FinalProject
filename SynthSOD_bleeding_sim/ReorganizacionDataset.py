import os
import shutil

# Rutas principales
inputs_dir = r"E:\DataBase_Demucs\inputs5"
outputs_dir = r"E:\DataBase_Demucs\OUTPUTSS\outputs5"

# Iterar sobre todas las carpetas en inputs
for folder in os.listdir(inputs_dir):
    input_path = os.path.join(inputs_dir, folder)
    output_path = os.path.join(outputs_dir, folder)
    
    if os.path.isdir(input_path) and os.path.isdir(output_path):
        destino_final = os.path.join(output_path, "inputs")
        
        # Si ya existe, eliminarla para evitar conflictos
        if os.path.exists(destino_final):
            shutil.rmtree(destino_final)
        
        # Copiar la carpeta entera
        shutil.copytree(input_path, destino_final)
        print(f"Copiado {input_path} → {destino_final}")

print("completado ")