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
    Procesa una imagen para detectar y segmentar una placa de vehículo utilizando Blackhat Morphological Operation.
    """
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    
    # Convertir a escala de grises
    imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar Blackhat para resaltar líneas oscuras sobre fondo claro
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 5))  # Kernel más ancho que alto
    blackhat = cv2.morphologyEx(imagen_gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Desenfocar la imagen para suavizar el ruido
    blackhat_blur = cv2.GaussianBlur(blackhat, (5, 5), 0)
    
    # Detectar bordes
    bordes = cv2.Canny(blackhat_blur, 60, 250)
    
    # Mostrar la imagen de bordes (opcional)
    # mostrar_imagen(bordes, "Bordes detectados")

    # Encontrar contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos para encontrar la placa
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 800:  # Ajustar según el tamaño esperado de la placa
            x, y, w, h = cv2.boundingRect(contorno)
            # Filtrar por relación de aspecto de una patente típica
            relacion_aspecto = w / h if h != 0 else 0
            if 2.0 <= relacion_aspecto <= 6.0:  # Relación de aspecto típica de placas
                placa_segmentada = imagen[y:y+h, x:x+w]
                mostrar_imagen(placa_segmentada, "Placa Segmentada")
                detectar_caracteres(placa_segmentada)
                break  # Salir después de encontrar la primera placa

def detectar_caracteres_conectados(placa):
    """
    Detecta y segmenta caracteres en una imagen de placa utilizando contornos y mejora del preprocesamiento.
    Dibuja rectángulos alrededor de los caracteres detectados.
    """
    # Convertir la imagen a escala de grises
    placa_gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    
    # Filtrado bilateral para suavizar la imagen y reducir el ruido
    placa_gray = cv2.bilateralFilter(placa_gray, 15, 75, 75)
    
    # Usar Canny para detectar los bordes
    edges = cv2.Canny(placa_gray, 100, 200)
    
    # Aplicar dilatación para mejorar los bordes
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Encontrar contornos en la imagen dilatada
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujar rectángulos alrededor de cada contorno encontrado
    for contour in contours:
        # Obtener el rectángulo delimitador de cada contorno
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filtrar contornos que sean demasiado pequeños (probablemente ruido)
        if w > 15 and h > 20:  # Ajustar estos valores según el tamaño de los caracteres
            cv2.rectangle(placa, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rectángulo verde

    # Mostrar la imagen con los caracteres detectados
    mostrar_imagen(placa, "Placa con Caracteres Detectados")

def procesar_imagenes_en_directorio(directorio):
    for archivo in os.listdir(directorio):
        if archivo.endswith(('.png', '.jpg', '.jpeg')):  # Filtrar por extensiones de imagen
            ruta_imagen = os.path.join(directorio, archivo)
            print(f"Procesando {ruta_imagen}...")
            procesar_imagen(ruta_imagen)

# Ruta del directorio con las imágenes
directorio_imagenes = "TP2/imagenes"
procesar_imagenes_en_directorio(directorio_imagenes)
