import os
from ultralytics import YOLO

def iniciar_entrenamiento():
    # 1. Inicializar el modelo YOLO Nano base (ligero y rápido)
    print("Cargando modelo base de YOLO...")
    modelo = YOLO("yolov8n.pt")

    # 2. Ruta del archivo de configuración en español
    ruta_dataset = "datos.yaml" 

    # Validación de seguridad: Verificar si el archivo de configuración existe
    if not os.path.exists(ruta_dataset):
        print(f"Error: No se encontró el archivo '{ruta_dataset}' en la raíz del proyecto.")
        print("Asegúrate de haber creado el archivo datos.yaml antes de continuar.")
        return

    print("Iniciando el proceso de entrenamiento de componentes eléctricos...")
    
    # 3. Lanzar el entrenamiento usando la CPU del contenedor
    # Nota: Usamos device='cpu' porque los entornos virtuales gratuitos no suelen tener GPU.
    modelo.train(
        data=ruta_dataset,   # Apunta a tu datos.yaml
        epochs=50,           # Cantidad de vueltas de entrenamiento
        imgsz=640,           # Tamaño estándar de imágenes para YOLO
        batch=16,            # Paquetes de imágenes por iteración
        device='cpu',        # Evita errores de falta de tarjeta gráfica dedicada
        name="entrenamiento_componentes_electricos" # Carpeta de salida
    )
    
    print("¡Entrenamiento finalizado con éxito!")
    print("El modelo final se ha guardado en la carpeta: runs/detect/entrenamiento_componentes_electricos/weights/best.pt")


    iniciar_entrenamiento()