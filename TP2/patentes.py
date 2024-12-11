import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagen(imagen, titulo, blocking=False):
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis('off')
    plt.show(block=blocking)

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
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    posibles_patentes = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 800:  # Ajustar según el tamaño esperado de la placa
            x, y, w, h = cv2.boundingRect(contorno)
            # Filtrar por relación de aspecto de una patente típica
            relacion_aspecto = w / h if h != 0 else 0
            if 2.0 <= relacion_aspecto <= 6.0:  # Relación de aspecto típica de placas
                posibles_patentes.append(imagen[y:y+h, x:x+w])
    
    return posibles_patentes, imagen

def encontrar_patente(posibles_patentes, img_original):
    """
    Detecta y segmenta caracteres en una imagen de placa utilizando contornos y mejora del preprocesamiento.
    """
    se_encontro_patente = False  # Bandera para indicar si se encontró una patente válida

    # Itera sobre todas las posibles placas detectadas
    for pat in posibles_patentes:
        gris = cv2.cvtColor(pat, cv2.COLOR_BGR2GRAY)  # Convertir la placa a escala de grises
        m = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))  # Definir un kernel morfológico
        eq_img = cv2.equalizeHist(gris)  # Ecualizar el histograma para mejorar el contraste
        img_black = cv2.morphologyEx(eq_img, cv2.MORPH_BLACKHAT, m, iterations=9)  # Aplicar la operación Blackhat
        _, img_bin = cv2.threshold(img_black, 70, 255, cv2.THRESH_BINARY_INV)  # Umbralizar la imagen
        
        # Detectar componentes conectados (posibles caracteres)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8)
        filtered_img = np.zeros_like(img_bin, dtype=np.uint8)

        # Filtrar componentes basados en área
        for i in range(1, num_labels):
            if 10 <= stats[i, cv2.CC_STAT_AREA] <= 100:
                filtered_img[labels == i] = 255
        
        img_draw = pat.copy()  # Copiar la imagen para dibujar los resultados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered_img, connectivity=8)

        # Calcular estadísticas de los componentes detectados (altura, ancho, etc.)
        hs = stats[1:, cv2.CC_STAT_HEIGHT]
        median_altura = np.median(hs)
        ys = stats[1:, cv2.CC_STAT_TOP]
        median_y = np.median(ys)
        anchos = stats[1:, cv2.CC_STAT_WIDTH]
        mediana_anchos = np.median(anchos)

        caracteres = []  # Lista para almacenar los caracteres detectados
        caracteres_coordenadas = []  # Lista para almacenar las coordenadas de los caracteres

        # Filtrar los componentes para detectar caracteres relevantes
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w < 4 or h < 6:  # Ignorar componentes demasiado pequeños
                continue
            if mediana_anchos - 4 < w < mediana_anchos + 10 and median_altura - 7 < h < median_altura + 5 and median_y - 15 < y < median_y + 15:
                caracteres.append(pat[0:pat.shape[0], x:x+w])  # Extraer el caracter
                caracteres_coordenadas.append((x, pat[0:pat.shape[0], x:x+w]))  # Guardar las coordenadas
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))  # Generar un color aleatorio para dibujar
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 1)  # Dibujar el contorno del caracter
        
        # Si se detectaron exactamente 6 caracteres (como es típico en las placas), organizarlos
        if len(caracteres) == 6:
            caracteres_ordenados = sorted(caracteres_coordenadas, key=lambda char: char[0])  # Ordenar los caracteres por su posición en el eje x
            caracteres = [char[1] for char in caracteres_ordenados]  # Extraer solo los caracteres ordenados
            altura_caracteres = caracteres[0].shape[0]  # Obtener la altura de los caracteres
            espacio_img = np.ones((altura_caracteres, 4, 3), dtype=np.uint8) * 255  # Espacio blanco entre los caracteres
            
            fila_caracteres = []  # Lista para organizar los caracteres en una fila
            for i, char in enumerate(caracteres):
                fila_caracteres.append(char)
                if i < len(caracteres) - 1:
                    fila_caracteres.append(espacio_img)  # Añadir espacio entre los caracteres
            
            fila_caracteres = np.hstack(fila_caracteres)  # Crear la fila de caracteres
            ancho_fila = fila_caracteres.shape[1]  # Obtener el ancho de la fila de caracteres
            pat_resized = cv2.resize(pat, (ancho_fila, pat.shape[0]), interpolation=cv2.INTER_LINEAR)  # Redimensionar la placa para ajustarse a la fila
            espacio_patente_caracteres = np.ones((3, pat_resized.shape[1], 3), dtype=np.uint8) * 255  # Espacio blanco entre la placa y los caracteres
            imagen_final = np.vstack([pat_resized, espacio_patente_caracteres, fila_caracteres])  # Combinar la placa y los caracteres en una sola imagen

            mostrar_imagen(img_original, 'Imagen recibida', blocking=True)  # Mostrar la imagen original
            mostrar_imagen(imagen_final, 'Patente y caracteres detectados', blocking=True)  # Mostrar la imagen con la patente y los caracteres detectados
            se_encontro_patente = True  # Indicar que se encontró una patente
    
    return se_encontro_patente  # Retornar si se encontró o no una patente

def procesar_imagenes_en_directorio(directorio):
    for archivo in os.listdir(directorio):
        if archivo.endswith(('.png', '.jpg', '.jpeg')):
            ruta_imagen = os.path.join(directorio, archivo)
            print(f"Procesando {ruta_imagen}...")
            posibles_patentes, img_original = procesar_imagen(ruta_imagen)
            encontrar_patente(posibles_patentes, img_original)

# Ruta del directorio con las imágenes
directorio_imagenes = "TP2/imagenes"
procesar_imagenes_en_directorio(directorio_imagenes)