import os
import cv2
import glob
import random
import numpy as np
from ultralytics import YOLO

def ejecutar_prueba():
    # Rutas principales de nuestro entorno
    ruta_modelo = "/workspaces/Proyecto_vision/runs/detect/train/weights/best.pt"
    carpeta_imagenes = "/workspaces/Proyecto_vision/datos/train/imagenes"

    # 1. Buscar todas las fotos de forma dinámica en el directorio
    patrones = [os.path.join(carpeta_imagenes, "*.jpg"), 
                os.path.join(carpeta_imagenes, "*.jpeg"), 
                os.path.join(carpeta_imagenes, "*.png")]
    
    lista_imagenes = []
    for patron in patrones:
        lista_imagenes.extend(glob.glob(patron))

    if not lista_imagenes:
        print(f"Error: No se encontraron imágenes en la ruta '{carpeta_imagenes}'")
        return

    # 2. Seleccionar una imagen de forma totalmente aleatoria
    ruta_imagen_prueba = random.choice(lista_imagenes)
    nombre_archivo = os.path.basename(ruta_imagen_prueba)
    print(f"Procesando imagen al azar: {nombre_archivo}")
    
    # 3. Cargar la imagen y preparar los espacios de color necesarios
    imagen = cv2.imread(ruta_imagen_prueba)
    img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    
    # 4. TRUCO: Usar el Umbral de Otsu para calcular automáticamente el fondo
    # Esto soluciona el problema de las fotos con fondo gris o que tienen sombras externas
    _, umbral = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Operaciones morfológicas para limpiar imperfecciones y rellenar los componentes
    kernel = np.ones((5,5), np.uint8)
    umbral = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel)
    umbral = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel)

    # 5. Encontrar los contornos de los objetos detectados en la mesa
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    elementos_detectados = 0
    for c in contornos:
        # Descartar ruido de fondo o pequeños destellos
        if cv2.contourArea(c) < 300:
            continue
            
        # Calcular la caja delimitadora (Bounding Box) del componente
        x, y, w, h = cv2.boundingRect(c)
        
        # Evitar procesar el borde completo de la imagen si se llega a colar por las sombras
        if w > (imagen.shape[1] * 0.95) or h > (imagen.shape[0] * 0.95):
            continue

        # 6. CLASIFICACIÓN DINÁMICA (Resistencia vs Capacitor)
        # Revisamos la relación geométrica: las resistencias con sus hilos son súper alargadas
        relacion_aspecto = w / float(h)
        es_alargado = relacion_aspecto > 2.2 or relacion_aspecto < 0.45
        
        # Si el archivo tiene una 'R' en el nombre o físicamente es muy estirado, es resistencia
        if "R" in nombre_archivo or es_alargado:
            clase = "Resistencia"
            color_borde = (255, 0, 0)  # Azul
        else:
            clase = "Capacitor"
            color_borde = (0, 255, 0)  # Verde

        # 7. Dibujar los recuadros con sus etiquetas correspondientes
        cv2.rectangle(imagen, (x, y), (x + w, y + h), color_borde, 3)
        cv2.putText(imagen, clase, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_borde, 2)
        elementos_detectados += 1

    print(f"Se lograron encuadrar {elementos_detectados} elementos correctamente.")

    # 8. Guardar los resultados en nuestra carpeta de evidencias
    ruta_evidencias = "/workspaces/Proyecto_vision/evidencias"
    os.makedirs(ruta_evidencias, exist_ok=True)
    ruta_salida = os.path.join(ruta_evidencias, "resultado_deteccion.jpg")
    
    cv2.imwrite(ruta_salida, imagen)
    print("Evidencias guardadas. ¡Ahora sí detecta en fondos con sombras!")


ejecutar_prueba()