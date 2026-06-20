import os
from ultralytics import YOLO

def iniciar_entrenamiento():
    # Cargamos el modelo base de YOLO (la versión Nano, que es ligera para la CPU)
    print("Cargando modelo base de YOLO...")
    modelo = YOLO("yolov8n.pt")

    # Archivo de configuración con las rutas de las imágenes y las clases
    ruta_dataset = "datos.yaml" 

    # Validación rápida para no arrancar a ciegas si falta el archivo .yaml
    if not os.path.exists(ruta_dataset):
        print(f"Error: No se encontró el archivo '{ruta_dataset}' en la raíz.")
        return

    print("Iniciando el proceso de entrenamiento...")
    
    # Arrancamos el entrenamiento con los parámetros del proyecto
    modelo.train(
        data=ruta_dataset,   
        epochs=10,           # Ajustado a 10 épocas para el avance actual
        imgsz=640,           # Tamaño de imagen estándar para YOLO
        device='cpu',        # Forzamos uso de CPU para entorno Codespaces
        name="entrenamiento"  # Define el nombre de la carpeta de salida en 'runs/'
    )
    
    print("¡Entrenamiento finalizado con éxito!")

    iniciar_entrenamiento()