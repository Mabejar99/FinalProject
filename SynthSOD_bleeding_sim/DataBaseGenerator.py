import os
import shutil
import re
from pydub import AudioSegment
from pydub.utils import which
import subprocess

# Configura tus rutas
carpeta1 = "D:\\FuentesAudio\\musdb18hq\\train\\3"
carpeta2 = "D:\\DataBase_Demucs\\VACIO"  # 
salida = "D:\\DataBase_Demucs\\inputs2"

# Ruta a ffmpeg y ffprobe
AudioSegment.converter = r"C:\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"
AudioSegment.ffprobe = r"C:\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

# MODIFICACION PARA SACAR BBDD CON 2-3-4 CANALES
usar_solo_una_carpeta = True

# Subcarpetas de carpeta1
subcarpetas1 = sorted([os.path.join(carpeta1, d) for d in os.listdir(carpeta1) if os.path.isdir(os.path.join(carpeta1, d))])

# Subcarpetas de carpeta2 (si se usa)
if usar_solo_una_carpeta:
    subcarpetas2 = [None]
else:
    subcarpetas2 = sorted([os.path.join(carpeta2, d) for d in os.listdir(carpeta2) if os.path.isdir(os.path.join(carpeta2, d))])

# Crear carpeta de salida
os.makedirs(salida, exist_ok=True)

contador = 1
limite = 50  #

for sub2 in subcarpetas2:
    nombre2 = os.path.basename(sub2) if sub2 else "NINGUNA"

    for sub1 in subcarpetas1:
        if contador > limite:
            break

        nombre1 = os.path.basename(sub1)
        destino_comb = os.path.join(salida, f"combinacion{contador}")
        os.makedirs(destino_comb, exist_ok=True)

        info_txt = [
            "Subcarpetas combinadas:",
            f"- Desde Carpeta1: {nombre1}",
            f"- Desde Carpeta2: {nombre2}" if sub2 else " Solo se usó Carpeta1"
        ]

        # Procesar carpeta1
        for archivo in os.listdir(sub1):
            origen = os.path.join(sub1, archivo)
            if os.path.isfile(origen):
                nombre, ext = os.path.splitext(archivo)
                destino = os.path.join(destino_comb, f"{nombre}.flac")
                os.chmod(origen, 0o777)
                try:
                    command = [
                        'ffmpeg', '-ss', '15',
                        '-i', origen, '-t', '30',
                        '-ac', '1',
                        '-sample_fmt', 's16',
                        '-ar', '44100',
                        '-acodec', 'flac',
                        '-y',
                        destino
                    ]
                    result = subprocess.run(command, capture_output=True, text=True, check=True)
                    print("Salida estándar:", result.stdout)
                    print("Salida de error:", result.stderr)
                    if os.path.exists(destino):
                        print(f" Archivo de carpeta1 convertido: {destino}")
                    else:
                        print(f" No se creó el archivo {destino}")
                except subprocess.CalledProcessError as e:
                    print(f" Error al convertir archivo de carpeta1: {e}")

        # Procesar carpeta2 (solo si existe)
        if sub2:
            for archivo in os.listdir(sub2):
                origen = os.path.join(sub2, archivo)
                if os.path.isfile(origen):
                    nombre_base, extension = os.path.splitext(archivo)
                    match = re.search(r"\(([^)]+)\)", nombre_base)
                    if match:
                        instrumento = match.group(1).strip()
                        if " - " in instrumento:
                            instrumento = instrumento.split(" - ", 1)[1].strip()
                    else:
                        instrumento = nombre_base.strip()

                    nombre_base_final = instrumento
                    nuevo_nombre = f"{nombre_base_final}.flac"
                    nombres_existentes = [
                        os.path.splitext(f)[0].lower()
                        for f in os.listdir(destino_comb)
                        if os.path.isfile(os.path.join(destino_comb, f))
                    ]

                    contador_archivo = 2
                    nombre_base_final_temp = nombre_base_final

                    while nombre_base_final_temp.lower() in nombres_existentes:
                        nombre_base_final_temp = f"{instrumento}{contador_archivo}"
                        contador_archivo += 1

                    nuevo_nombre = f"{nombre_base_final_temp}.flac"
                    destino = os.path.join(destino_comb, nuevo_nombre)
                    os.chmod(origen, 0o777)
                    try:
                        command = [
                            'ffmpeg', '-ss', '15',
                            '-i', origen, '-t', '30',
                            '-ac', '1',
                            '-sample_fmt', 's16',
                            '-ar', '44100',
                            '-acodec', 'flac',
                            '-y',
                            destino
                        ]
                        result = subprocess.run(command, capture_output=True, text=True, check=True)
                        print("Salida estándar:", result.stdout)
                        print("Salida de error:", result.stderr)
                        if os.path.exists(destino):
                            print(f" Archivo de carpeta2 convertido: {destino}")
                        else:
                            print(f" No se creó el archivo {destino}")
                    except subprocess.CalledProcessError as e:
                        print(f" Error al convertir archivo de carpeta2: {e}")

        # Escribir info.txt
        ruta_info = os.path.join(destino_comb, "info.txt")
        with open(ruta_info, "w", encoding="utf-8") as f:
            f.write("\n".join(info_txt))

        # Borrar archivos no deseados
        for borrar in ["mixture.flac", "other.flac", "bass.flac"]:
            ruta_borrar = os.path.join(destino_comb, borrar)
            if os.path.exists(ruta_borrar):
                os.remove(ruta_borrar)
                print(f" Archivo eliminado: {borrar} en {destino_comb}")
        


        contador += 1

print(f"\n completo: generadas {contador - 1} combinaciones.")

