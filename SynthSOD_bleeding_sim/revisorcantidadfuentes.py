import os

# Cambia esta ruta por la carpeta raíz que quieras analizar
carpeta_raiz = "D:/DataBase_Demucs/inputs"

# Recorre todas las subcarpetas
for ruta_actual, subcarpetas, archivos in os.walk(carpeta_raiz):
    cantidad_archivos = len([f for f in archivos if os.path.isfile(os.path.join(ruta_actual, f))])
    if cantidad_archivos == 13:
        print(f"La carpeta '{ruta_actual}' tiene exactamente 13 archivos.")
    else:
        print(f"La carpeta '{ruta_actual}' tiene {cantidad_archivos} archivos. ")
