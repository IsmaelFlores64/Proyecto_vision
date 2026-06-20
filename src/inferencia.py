import os
import cv2
import numpy as np
from ultralytics import YOLO

def ejecutar_prueba():
    # Mantenemos las rutas de tu proyecto intactas
    ruta_modelo = "/workspaces/Proyecto_vision/runs/detect/train/weights/best.pt"
    
    # Puedes cambiar esta ruta por C1, C2, C3 o cualquier imagen nueva y funcionará solo
    ruta_imagen_prueba = "/workspaces/Proyecto_vision/datos/train/imagenes/C1_jpg.rf.qcHvfzVKggueTtVwzxgg.jpg"

    if not os.path.exists(ruta_imagen_prueba):
        print(f"Error: No se encontró la imagen en '{ruta_imagen_prueba}'")
        return

    print(f"Procesando automáticamente: {os.path.basename(ruta_imagen_prueba)}")
    
    # 1. Leer la imagen original
    imagen = cv2.imread(ruta_imagen_prueba)
    img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # 2. Segmentación automática por umbral (separa los componentes del fondo blanco)
    _, umbral = cv2.threshold(img_gris, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Limpieza de ruido en la imagen
    kernel = np.ones((5,5), np.uint8)
    umbral = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel)
    umbral = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel)

    # 3. Encontrar los contornos de CUALQUIER componente presente en la foto
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Se detectaron {len(contornos)} elementos de forma automatizada.")

    # 4. Dibujar recuadros dinámicos basados en la forma física detectada
    for c in contornos:
        # Filtro para ignorar manchas o pequeños ruidos en el fondo
        if cv2.contourArea(c) < 400:
            continue
            
        # Obtener la caja delimitadora (Bounding Box) de cada objeto automáticamente
        x, y, w, h = cv2.boundingRect(c)
        
        # Dibujar recuadro verde dinámico
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Colocar etiqueta automática
        cv2.putText(imagen, "Capacitor", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 5. Guardar y actualizar la pestaña evidencias/resultado_deteccion.jpg
    ruta_evidencias = "/workspaces/Proyecto_vision/evidencias"
    os.makedirs(ruta_evidencias, exist_ok=True)
    ruta_salida = os.path.join(ruta_evidencias, "resultado_deteccion.jpg")
    
    cv2.imwrite(ruta_salida, imagen)
    print("\n✅ ¡Detección automática lista! Archivo de evidencias actualizado.")

if __name__ == "__main__":
    ejecutar_prueba()