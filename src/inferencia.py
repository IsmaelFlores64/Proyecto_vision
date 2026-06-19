import os
from ultralytics import YOLO

def ejecutar_prueba():
    # 1. Definir la ruta del modelo entrenado
    # Apunta a la carpeta por defecto donde YOLO deja los resultados del entrenamiento
    ruta_modelo = "runs/detect/entrenamiento_componentes_electricos/weights/best.pt"
    
    if not os.path.exists(ruta_modelo):
        print(f"Error: No se encuentra el modelo entrenado en '{ruta_modelo}'.")
        print("Primero debes ejecutar exitosamente el script de entrenamiento.py")
        return

    # 2. Cargar el modelo YOLO con tus pesos entrenados
    modelo = YOLO(ruta_modelo)
    print("Modelo entrenado cargado exitosamente.")

    # 3. Ruta de la imagen que quieres probar (la que subiste a entrenamiento u otra de prueba)
    ruta_imagen_prueba = "datos/imagenes/entrenamiento/resistencia_1.jpg"
    
    if not os.path.exists(ruta_imagen_prueba):
        print(f"Error: No se encontró la imagen de prueba en '{ruta_imagen_prueba}'")
        return

    # 4. Ejecutar la predicción
    print(f"Procesando detección en: {ruta_imagen_prueba}...")
    resultados = modelo(ruta_imagen_prueba)

    # 5. Guardar el resultado visual en la carpeta de evidencias exigida
    # Aseguramos que la carpeta de evidencias exista
    os.makedirs("evidencias", exist_ok=True)
    
    for r in resultados:
        # save=True guarda automáticamente la imagen pintada con los recuadros
        # project y name definen que se guarde exactamente dentro de tu carpeta 'evidencias'
        r.save(filename="evidencias/resultado_deteccion.jpg")
        
    print("¡Detección completada!")
    print("El resultado con los componentes marcados se guardó en: evidencias/resultado_deteccion.jpg")


ejecutar_prueba();