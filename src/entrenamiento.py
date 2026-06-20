import os
from ultralytics import YOLO

def iniciar_entrenamiento():
    print("Cargando modelo base original...")
    # Cargamos el modelo base desde la raíz
    modelo = YOLO("/workspaces/Proyecto_vision/yolov8n.pt")

    print("Iniciando entrenamiento real (10 épocas)...")
    modelo.train(
        data="/workspaces/Proyecto_vision/dataset.yaml",   
        epochs=10,           # 10 épocas como pediste
        imgsz=640,           
        device='cpu',        
        exist_ok=True,       # Sobrescribe la carpeta 'train' existente
        project="/workspaces/Proyecto_vision/runs/detect",
        name="train"
    )
    print("¡Entrenamiento finalizado con éxito!")

if __name__ == "__main__":
    iniciar_entrenamiento()