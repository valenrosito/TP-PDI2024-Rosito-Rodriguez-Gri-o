import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def mostrar_imagen(imagen, titulo):
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis('off')
    plt.show()

def procesar_imagen(ruta_imagen):
    """
    Procesa una imagen para detectar y segmentar una placa de vehículo.
    Args:
        ruta_imagen (str): La ruta del archivo de imagen a procesar.
    Pasos:
        1. Cargar la imagen desde la ruta especificada.
        2. Desenfocar la imagen utilizando un filtro Gaussiano.
        3. Convertir la imagen desenfocada a escala de grises.
        4. Detectar bordes en la imagen en escala de grises.
        5. Encontrar contornos en la imagen de bordes.
        6. Filtrar los contornos para encontrar la placa del vehículo basada en el área del contorno.
        7. Si se encuentra una placa, segmentar la región de la placa y mostrarla.
        8. Detectar caracteres en la placa segmentada.
    Nota:
        El área mínima del contorno para considerar una placa es de 1500 píxeles, 
        este valor puede ajustarse según el tamaño esperado de la placa.
    """

    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    
    # Desenfocar la imagen
    imagen_blur = cv2.GaussianBlur(imagen, (5, 5), 0)
    
    # Convertir a escala de grises
    imagen_gray = cv2.cvtColor(imagen_blur, cv2.COLOR_BGR2GRAY)
    
    # Detectar bordes
    bordes = cv2.Canny(imagen_gray, 50, 250)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos para encontrar la placa
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 1500:  # Ajustar según el tamaño esperado de la placa
            x, y, w, h = cv2.boundingRect(contorno)
            placa_segmentada = imagen[y:y+h, x:x+w]
            mostrar_imagen(placa_segmentada, "Placa Segmentada")
            detectar_caracteres(placa_segmentada)
            break  # Salir después de encontrar la primera placa

def detectar_caracteres(placa):
    """
    Detecta y segmenta caracteres en una imagen de placa de vehículo.
    Args:
        placa (numpy.ndarray): Imagen de la placa en formato BGR.
    Returns:
        None: La función no retorna ningún valor. Los caracteres detectados se dibujan directamente sobre la imagen de entrada.
    El proceso incluye:
        1. Convertir la imagen a escala de grises.
        2. Aplicar umbralización con Otsu para binarizar la imagen.
        3. Encontrar contornos en la imagen binarizada.
        4. Filtrar contornos por tamaño y relación de aspecto para identificar caracteres.
        5. Dibujar rectángulos alrededor de los caracteres detectados.
        6. Agrupar caracteres en grupos de 3 y dibujar rectángulos alrededor de cada grupo.
        7. Mostrar la imagen resultante con los caracteres recuadrados.
    """
    # Convertir a escala de grises
    placa_gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    
    # Umbral con Otsu
    _, imagen_binaria = cv2.threshold(placa_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    caracteres_segmentados = []
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        relacion_aspecto = h / w if w != 0 else 0
        
        # Filtrar por tamaño y relación de aspecto
        if w > 10 and h > 20 and 1.5 <= relacion_aspecto <= 3.0:
            caracteres_segmentados.append((x, y, w, h))
            # Dibujar rectángulo alrededor del carácter
            cv2.rectangle(placa, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Agrupar caracteres en grupos de 3
    for i in range(0, len(caracteres_segmentados), 3):
        grupo = caracteres_segmentados[i:i + 3]
        if len(grupo) == 3:
            # Dibujar rectángulo alrededor del grupo
            x1, y1, w1, h1 = grupo[0]
            x2, y2, w2, h2 = grupo[2]
            cv2.rectangle(placa, (x1, y1), (x2 + w2, y1 + h1), (255, 0, 0), 2)

    mostrar_imagen(placa, "Placa con Caracteres Recuadrados")


def procesar_imagenes_en_directorio(directorio):
    for archivo in os.listdir(directorio):
        if archivo.endswith(('.png', '.jpg', '.jpeg')):  # Filtrar por extensiones de imagen
            ruta_imagen = os.path.join(directorio, archivo)
            print(f"Procesando {ruta_imagen}...")
            procesar_imagen(ruta_imagen)


directorio_imagenes = "TP2/imagenes"
procesar_imagenes_en_directorio(directorio_imagenes)
