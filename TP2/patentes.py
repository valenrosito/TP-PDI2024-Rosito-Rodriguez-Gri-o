
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
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    
    # Desenfocar la imagen
    imagen_blur = cv2.GaussianBlur(imagen, (5, 5), 0)
    
    # Convertir a escala de grises
    imagen_gray = cv2.cvtColor(imagen_blur, cv2.COLOR_BGR2GRAY)
    
    # Detectar bordes
    bordes = cv2.Canny(imagen_gray, 30, 200)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos para encontrar la placa
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 700:  # Ajustar según el tamaño esperado de la placa
            x, y, w, h = cv2.boundingRect(contorno)
            # Dibujar rectángulo alrededor de la patente
            cv2.rectangle(imagen, (x, y), (x + w, y + h), (255, 0, 0), 2)
            placa_segmentada = imagen[y:y+h, x:x+w]
            mostrar_imagen(placa_segmentada, "Placa Segmentada")
            detectar_caracteres(placa_segmentada)
            break  # Salir después de encontrar la primera placa

def detectar_caracteres(placa):
    # Convertir a escala de grises
    placa_gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    
    # Umbral
    _, imagen_binaria = cv2.threshold(placa_gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    caracteres_segmentados = []
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        relacion_aspecto = h / w if w != 0 else 0
        
        # Filtrar por tamaño y relación de aspecto
        if w > 5 and h > 15 and 1.5 <= relacion_aspecto <= 3.0:
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