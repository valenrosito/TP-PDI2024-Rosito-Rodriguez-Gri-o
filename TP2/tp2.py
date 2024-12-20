import matplotlib.pyplot as plt
import numpy as np
import cv2

def segmentacion_monedas_dados(ruta_imagen):
    """
    Segmenta monedas y dados en una imagen dada.

    Args:
        ruta_imagen (str): La ruta del archivo de la imagen a procesar.

    Returns:
        tuple: Una tupla que contiene:
            - monedas (list): Una lista de tuplas, cada una con un contorno y el área de una moneda detectada.
            - mascara (numpy.ndarray): Una imagen binaria con los dados segmentados.
    """
    # Cargamos la imagen y le aplicamos un filtro de desenfoque para eliminar el ruido
    image = cv2.imread(ruta_imagen)
    image = cv2.blur(image, (7, 7))

    # Convertimos la imagen a hls y aplicamos sobre el canal de saturacion un filtro de desenfoque 
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    image_blureada = cv2.blur(s, (3, 3))

    # Aplicamos un umbral para obtener una imagen binaria y poder realizar operaciones morfologicas
    _, imagen_binaria = cv2.threshold(image_blureada, 12, 255, cv2.THRESH_BINARY)


    s = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    close = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, s)
    open_ = cv2.morphologyEx(close, cv2.MORPH_OPEN, s)

    s = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    erode = cv2.erode(open_, s)


    contornos, _ = cv2.findContours(erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mascara = np.zeros_like(erode, dtype=np.uint8)

    monedas = []
    dados = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        fp = area / (perimetro**2)
        if fp > 0.06:
            monedas.append((contorno, area))
        else:
            cv2.drawContours(mascara, [contorno], -1, 255, thickness=cv2.FILLED)


    return monedas, mascara

def conteo_monedas(monedas):
    """
    Cuenta la cantidad de monedas de diferentes denominaciones en una lista dada.

    Args:
        monedas (list): Una lista de tuplas, donde cada tupla contiene dos elementos:
                        - El primer elemento es un identificador de la moneda (puede ser cualquier tipo de dato).
                        - El segundo elemento es el área de la moneda (un número entero o flotante).

    Returns:
        None: La función imprime la cantidad de monedas de 10 centavos, 50 centavos y 1 peso encontradas en la lista.
    """
    # Ordenamos la lista de monedas en base a su area para facilitar la visualziacion
    sorted_monedas = sorted(monedas , key=lambda x: x[1])
    ten_cents = 0
    fifty_cents = 0
    one_peso = 0

    for moneda in sorted_monedas:
        if moneda[1] > 95000:
            fifty_cents += 1
        elif moneda[1] < 65000:
            ten_cents += 1
        elif moneda[1] < 85000 and moneda[1] > 65000:
            one_peso += 1

    return print(f"Se encontraron las siguientes cantidades de monedas:\nMonedas 10 centavos: {ten_cents}\nMonedas 50 centavos: {fifty_cents}\nMonedas 1 peso: {one_peso}")

def conteo_dados(mascara):
    """
    Cuenta el número de puntos en los dados en una imagen.

    Esta función procesa una imagen de máscara de entrada para identificar y contar el número de puntos en cada dado presente en la imagen.
    Realiza varias operaciones morfológicas para limpiar la imagen y aislar los dados, luego extrae cada dado y cuenta el número de puntos en él.

    Args:
        mascara (numpy.ndarray): La imagen de máscara de entrada donde se detectarán los dados.

    Returns:
        None: La función imprime el número de puntos en cada dado encontrado en la imagen.
    """
    imagen_nueva = cv2.imread('TP2/monedas.jpg', cv2.IMREAD_GRAYSCALE)

    matrix = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    close = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, matrix)

    matrix = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    erode = cv2.morphologyEx(close, cv2.MORPH_ERODE, matrix)

    matrix = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
    open_ = cv2.morphologyEx(erode, cv2.MORPH_OPEN, matrix)

    dados, _ = cv2.findContours(open_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dados_recortados = []
    for dado in dados:
        contorno = dado
        cv2.drawContours(imagen_nueva, [contorno], -1, (0, 255, 0), 2)
        imagen_nueva = cv2.blur(imagen_nueva, (9, 9))

        x, y, w, h = cv2.boundingRect(contorno)
        recorte = imagen_nueva[y:y+h, x:x+w-50]
        dados_recortados.append(recorte)

        _, thresh = cv2.threshold(recorte, 145, 255, cv2.THRESH_BINARY)

        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        puntos = [cnt for cnt in contornos if cv2.contourArea(cnt) > 50]

        num_puntos = len(puntos)

        print(f"Dado {(len(dados_recortados)-1)+1} tiene {num_puntos} puntos")

conteo_monedas(segmentacion_monedas_dados("TP2/monedas.jpg")[0])
conteo_dados((segmentacion_monedas_dados("TP2/monedas.jpg")[1]))
