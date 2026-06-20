# Proyecto de Visión Artificial: Detección de Componentes Electrónicos

## Instrucciones para Ejecutar el Proyecto
1. **Clonar el repositorio**:
git clone https://github.com/IsmaelFlores64/Proyecto_vision.git
cd Proyecto_vision
2. **Instalar las dependencias necesarias**:
pip install ultralytics opencv-python matplotlib
3. **Ejecutar el entrenamiento del modelo**:
python entrenamiento.py
4. **Ejecutar la detección**:
python inferencia.py
5. **Visualizar los resultados**

Los resultados de las detecciones se almacenarán en la carpeta runs/detect/, donde se podrán observar las imágenes procesadas con los componentes identificados.

## Integrantes

* Jesús Ismael Flores Pardo (23310364)
* Matías Garay Mendoza (23310331)

## Estudio de Caso

### Problema a Resolver

En la industria electrónica, laboratorios y entornos educativos se utilizan una gran variedad de componentes electrónicos como resistencias, capacitores, LEDs, transistores e integrados. La identificación manual de estos componentes puede resultar lenta y propensa a errores, especialmente para estudiantes o personas con poca experiencia.

Este proyecto busca ser capaz de detectar e identificar componentes electrónicos de forma automática y en tiempo real.

### Descripción del Hardware Propuesto

Para implementar este sistema en un entorno real se propone el siguiente hardware:

* **Cámara HD o cámara industrial:** encargada de capturar imágenes o video de los componentes electrónicos.
* **Monitor o pantalla:** muestra el resultado de la detección, indicando el nombre y ubicación de cada componente.
* **Banda transportadora:** permite mover los componentes para su inspección automática.
* **Brazo robótico o mecanismo de clasificación:** recibe la información generada por el modelo y separa los componentes según su categoría.

### Flujo de Funcionamiento

El sistema utiliza una cámara para capturar imágenes de los componentes electrónicos que se encuentran sobre una mesa de trabajo o una banda transportadora. Estas imágenes son enviadas a una computadora o dispositivo de procesamiento donde se ejecuta el modelo YOLO previamente entrenado. El modelo analiza cada imagen en tiempo real, identifica los componentes electrónicos presentes y genera cuadros delimitadores alrededor de ellos junto con su nombre. Posteriormente, los resultados se muestran en una pantalla para que el usuario pueda visualizar fácilmente qué componentes fueron detectados. En una aplicación industrial, esta información también podría enviarse a un brazo robótico o sistema automatizado encargado de clasificar, organizar o separar los componentes según su tipo.

### Aplicación en la Vida Real

Este sistema podría utilizarse en laboratorios de electrónica, almacenes de componentes, instituciones educativas y líneas de producción electrónica. Su implementación permitiría agilizar procesos de identificación y clasificación, reducir errores humanos y servir como herramienta de apoyo para estudiantes que están aprendiendo a reconocer componentes electrónicos.

