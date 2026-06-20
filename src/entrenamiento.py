import os
from ultralytics import YOLO

def iniciar_entrenamiento():
    print("Cargando modelo base original...")
    # Cargar la arquitectura limpia de YOLOv8 para empezar el entrenamiento
    modelo = YOLO("/workspaces/Proyecto_vision/yolov8n.pt")

    print("Iniciando entrenamiento real (10 épocas)...")
    # Configurar y arrancar el entrenamiento con nuestro dataset
    modelo.train(
        data="/workspaces/Proyecto_vision/dataset.yaml",   # Archivo de configuración con las rutas de las imágenes
        epochs=10,                                         # Cantidad de ciclos completos de entrenamiento
        imgsz=640,                                         # Tamaño estándar al que se reescalan las imágenes
        device='cpu',                                      # Forzar el uso del procesador para el entrenamiento
        exist_ok=True,                                     # Evita que se creen carpetas duplicadas (train2, train3) y sobrescribe la actual
        project="/workspaces/Proyecto_vision/runs/detect", # Carpeta principal donde se guardan los resultados
        name="train"                                       # Nombre de la subcarpeta específica para este entrenamiento
    )
    print("¡Entrenamiento finalizado con éxito!")

    iniciar_entrenamiento()