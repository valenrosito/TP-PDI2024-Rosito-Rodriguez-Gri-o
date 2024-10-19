import cv2
import numpy as np
import matplotlib.pyplot as plt
# imagen = cv2.imread('examen_1.png', cv2.IMREAD_GRAYSCALE) 
# nombre = imagen[0:30, 60:255]
# fecha = imagen[0:30, 290:375]
# clase = imagen[0:30, 420:490] 
# plt.imshow(clase, cmap='gray')
# plt.show()


def detectar_lineas_verticales(imagen, umbral=int):
    _, img_th = cv2.threshold(imagen, 128, 255, cv2.THRESH_BINARY_INV)
    # Convertimos la imagen binaria a 1 y 0 (Para facilitar la sumas)
    img_th_ones = img_th // 255
    
    img_cols = np.sum(img_th_ones, axis=0)
    th_col = np.max(img_cols) * umbral
    img_cols_th = img_cols > th_col
    lineas_v = []
    en_linea = False
    for i, val in enumerate(img_cols_th):
        if val and not en_linea:
            inicio_col = i
            en_linea = True
        elif not val and en_linea:
            fin_col = i
            lineas_v.append((inicio_col, fin_col))
            en_linea = False
    return lineas_v

def detectar_lineas_horizontales(imagen, umbral=int):
    _, img_th = cv2.threshold(imagen, 128, 255, cv2.THRESH_BINARY_INV)
    # Convertimos la imagen binaria a 1 y 0 (Para facilitar la sumas)
    img_th_ones = img_th // 255    
    img_fils = np.sum(img_th_ones, axis=1)
    th_fil = np.max(img_fils) * umbral
    imh_fils_th = img_fils > th_fil

    lineas_h = []
    en_linea = False
    for i, val in enumerate(imh_fils_th):
        if val and not en_linea:
            inicio_fil = i
            en_linea = True
        elif not val and en_linea:
            fin_fil = i
            lineas_h.append((inicio_fil, fin_fil))
            en_linea = False
    return lineas_h

def division_bloques(lineas_v, lineas_h,img):
    blocks = []
    for i in range(len(lineas_h) - 1):
        for j in range(len(lineas_v) - 1):
            if j == 1:  # Esto excluye la segunda columna (los números del medio)
                continue
            x1 = lineas_v[j][1] + 1
            x2 = lineas_v[j+1][0]- 1
            y1 = lineas_h[i][1] + 1
            y2 = lineas_h[i+1][0]-1
            # Recortamos el bloque
            block = img[y1:y2, x1:x2]
            blocks.append(block)
    return blocks[2:]

    # fig, axs = plt.subplots(len(blocks)//2, 2, figsize=(10, 10))
    # axs = axs.flatten()
    # for i, block in enumerate(blocks):
    #     axs[i].imshow(block, cmap='gray')
    #     axs[i].set_title(f'Bloque {i+1}')
    #     axs[i].axis('off'), axs[i].set_xticks([]), axs[i].set_yticks([])  # Oculta los ejes
    # plt.tight_layout()  
    # plt.show()

    # print(f'Se detectaron {len(blocks)} blocks.')


def detectar_linea_pregunta(imagen):
    ##Hacemos esto ya que la imagen binarizada no encuentra la linea con los contornos
    imagen_entrada = cv2.cvtColor(imagen.copy(), cv2.COLOR_GRAY2BGR)
    _, img_binaria = cv2.threshold(imagen, 128, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ancho = 0
    linea = (0, 0, 0, 0)

    for contorno in contornos:
        # Encontramos la linea
        x, y, w, h = cv2.boundingRect(contorno)
        if ancho < w:  
            ancho = w
            linea = (x, y, w, h-15)
    x, y, w, h = linea
    cv2.rectangle(imagen_entrada, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Dibujar los contornos en la imagen
    # cv2.drawContours(imagen_entrada, contornos, -1, (255, 0, 0), 1) 
    # plt.imshow(cv2.cvtColor(imagen_entrada, cv2.COLOR_BGR2RGB))
    # plt.show()
    return linea

def detectar_respuesta(imagen, rectangulo):
    blocardo = imagen.copy()
    x, y, w, h = rectangulo
    respuesta = blocardo[y+h:y, x:x+w]
    _, respuesta = cv2.threshold(respuesta, 128, 255, cv2.THRESH_BINARY)
    
    conectados = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(respuesta, conectados, cv2.CV_32S)
    
    # Invertimos para que los contornos de los huecos sean detectados
    _, bin_respuesta = cv2.threshold(respuesta, 128, 255, cv2.THRESH_BINARY_INV)
    
    contornos, hierarchy = cv2.findContours(bin_respuesta, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    num_hijos = 0
    letra_detectada = ""
    area_hijo = 0
    respuestas_detectadas = []
    # Verificar la jerarquía para contar los hijos del contorno externo (índice 0)
    if hierarchy is not None:
        hijo_actual = hierarchy[0][0][2]  # Obtener el primer hijo del contorno 0
        while hijo_actual != -1:
            num_hijos += 1
            # Calcular el área del contorno del hijo
            area_hijo = cv2.contourArea(contornos[hijo_actual])
            hijo_actual = hierarchy[0][hijo_actual][0]  # Ir al siguiente hermano
        
        if len(contornos) > 3:
            letra_detectada = ""
        elif num_hijos == 0:
            letra_detectada = "C"  # No tiene hijos
        elif num_hijos == 1:
            # Distinguir entre A y D en función del área del hijo
            if area_hijo > 35:  # Umbral 
                letra_detectada = "D"
            else:
                letra_detectada = "A"
        elif num_hijos == 2:
            letra_detectada = "B"  # Tiene dos hijos (dos huecos, como la letra B)
    rta_examen.append(letra_detectada)
    # return letra_detectada
    
def detectar_caracteres_encabezado(imagen_encabezado):
    imagen_encabezado = cv2.adaptiveThreshold(imagen_encabezado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Encontrar contornos en la imagen umbralizada
    contours, hierarchy = cv2.findContours(imagen_encabezado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Dibujar los contornos en una imagen en blanco
    contour_img = np.zeros_like(imagen_encabezado)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # Detecta los componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_encabezado, 8, cv2.CV_32S)
    stats = sorted(stats, key=lambda x: x[0])

    # Filtra los componentes cuya área es menor a 50 pixeles
    stats_filtrado = [s for s in stats if s[-1] > 20]
    

    umbral_distancia = 35
    espacios = 0

    repetidos = []

    for i in range(len(stats_filtrado)-1):
        # Obtiene las coordenadas x del componente actual y del siguiente
        x_actual = stats_filtrado[i][0]
        x_siguiente = stats_filtrado[i + 1][0]
        # Calcula la distancia horizontal entre los componentes
        distancia_horizontal = x_siguiente - x_actual

        # Si la distancia horizontal es mayor que cierto umbral intuimos que hay un espacio entre palabras.
        if distancia_horizontal >= umbral_distancia:
            # Incrementa el contador de espacios entre letras
            espacios += 1

    salida = {"Caracteres": len(stats_filtrado),"Espacios": espacios,"Palabras": espacios + 1}
    return salida

nombres = []
examenes = ['examen_1.png','examen_2.png','examen_3.png','examen_4.png', 'examen_5.png']
for examen in examenes:
    rta_examen = []
    img = cv2.imread(examen, 2)
    # Detectar líneas verticales y horizontales
    lineas_v = detectar_lineas_verticales(img, 0.6)
    lineas_h = detectar_lineas_horizontales(img, 0.5)
    blocks = division_bloques(lineas_v, lineas_h, img)
    lineas_h = lineas_h[1:]
    
    # Corto cada campo
    nombre = img[0:30, 50:255]
    fecha = img[0:30, 290:375]
    clase = img[0:30, 410:490]

    
    nombres.append(nombre)

    d_nombre = detectar_caracteres_encabezado(nombre)
    d_fecha = detectar_caracteres_encabezado(fecha)
    d_class = detectar_caracteres_encabezado(clase)
    
    print(d_nombre,d_fecha,d_class)
    
    
    
    
    
    
    
    
    # Nombre
    if d_nombre["Caracteres"] > 25 or d_nombre["Caracteres"] == 0 or  d_nombre["Palabras"] < 2:
        print("Nombre: Mal")
    else:
        print("Nombre: Ok")
    
    # Fecha
    if d_fecha["Caracteres"] !=8:
        print("fecha: Mal")
    else:
        print("fecha: Ok")
    
    # Class
    if d_class["Caracteres"] == 2:
        print("Code: Ok")
    else:
        print("Code: Mal")
    
    
    for block in blocks:
        linea_pregunta = detectar_linea_pregunta(block)
        detectar_respuesta(block, linea_pregunta)
    respuestas_correctas = ['C', 'B', 'A', 'D', 'B', 'B', 'A', 'B', 'D', 'D']
    nuevo_orden = [0, 1, 4, 2, 6, 5, 3, 8, 9, 7]
    # Reorganizamos la lista de las respuestas de cada examen
    respuestas_correctas_ordenadas = [rta_examen[i] for i in nuevo_orden]

    contador = 0
    for i in range(len(respuestas_correctas)):
        if respuestas_correctas[i] == rta_examen[i]:
            print(f'Pregunta {(i+1)}:', ' OK')
            contador += 1
        else:
            print(f'Pregunta {(i+1)}:', ' MAL')
    print('Puntaje: ',contador,'/10')
