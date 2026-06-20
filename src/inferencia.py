import os
from ultralytics import YOLO

def ejecutar_prueba():
    # 1. Cargamos tus propios pesos entrenados (no el modelo base genérico)
    ruta_modelo = "runs/detect/entrenamiento/weights/best.pt"
    
    if not os.path.exists(ruta_modelo):
        print(f"Error: No se encuentra tu modelo entrenado en '{ruta_modelo}'.")
        print("Asegúrate de correr primero entrenamiento.py para generar tus pesos.")
        return

    print("Cargando tus pesos personalizados...")
    modelo = YOLO(ruta_modelo)
    
    # 2. Ruta de tu imagen de resistencias
    ruta_imagen_prueba = "datos/Entrenamiento/Imagenes/resistencia.jpg"
    
    if not os.path.exists(ruta_imagen_prueba):
        print(f"Error: No se encontró la imagen en '{ruta_imagen_prueba}'")
        return

    print(f"Detectando componentes en: {ruta_imagen_prueba}...")
    
    # 3. Corremos la predicción con los parámetros de visualización activos
    # conf=0.25 filtra detecciones fantasmas; save=True fuerza a pintar los recuadros
    resultados = modelo.predict(
        source=ruta_imagen_prueba,
        conf=0.25,
        save=True,
        line_width=3
    )

    # 4. Aseguramos tu carpeta exigida de evidencias y guardamos el archivo final ahí
    os.makedirs("evidencias", exist_ok=True)
    
    for r in resultados:
        # Esto guarda la foto con los recuadros de tus resistencias pintados
        r.save(filename="evidencias/resultado_deteccion.jpg")
        
    print("\n¡Detección completada con éxito!")
    print("Revisa tu resultado con los recuadros en: evidencias/resultado_deteccion.jpg")

    ejecutar_prueba()