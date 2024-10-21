import cv2
import numpy as np
import matplotlib.pyplot as plt


def detectar_lineas_verticales(imagen, umbral:float):
    """"
    Esta funcion se encarga de encontrar las lineas verticales con una imagen y un umbral como parametros para determinar el grosor 
    de la linea, esta binariza la imagen y retorna una lista con todas las lineas verticales que pasen el umbral especificado
    """
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

def detectar_lineas_horizontales(imagen, umbral:float):
    """"
    Esta funcion se encarga de encontrar las lineas horizontales con una imagen y un umbral como parametros para determinar el grosor 
    de la linea, esta binariza la imagen y retorna una lista con todas las lineas verticales que pasen el umbral especificado
    """
    
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
    """ 
    Recibe una lista con las lineas verticales, una lista con las lineas horizonales y una imagen y retorna una lista con los bloques
    
    Esta funcion se encarga de dividir en bloques la imagen utilizando la lista de las lineas verticales, la lista de las lineas horizontales
    y una imagen, devuelve una lista con los bloques necesarios para su posterior analisis
    """
    
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
    blocks = blocks[2:]
    # fig, axs = plt.subplots(len(blocks)//2, 2, figsize=(10, 10))
    # axs = axs.flatten()
    # for i, block in enumerate(blocks):
    #     axs[i].imshow(block, cmap='gray')
    #     axs[i].set_title(f'Bloque {i+1}')
    #     axs[i].axis('off'), axs[i].set_xticks([]), axs[i].set_yticks([])  # Oculta los ejes
    # plt.tight_layout()  
    # plt.show()
    return blocks

def detectar_linea_pregunta(imagen):
    """ 
    Recibe una imagen, devuelve el recuadro de la respuesta.
    
    Esta funcion es la encargada de detectar cual es la linea dentro del bloque mencionado anteriormente y devuelve un recuadro con las 
    coordenadas de la linea y del elemento que tiene adentro si es que posee alguno
    """
    
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
    """ 
    Recibe una imagen y un recuadro con las coordenadas del mismo y agrega un elemento a la lista de las respuestas del examen. No devuelve nada.
    
    Esta funcion es la encargada de encontrar cual es el valor que encuentra en la linea de la respuesta. Puede ser A,B,C,D o ""
    """
    
    blocardo = imagen.copy()
    x, y, w, h = rectangulo
    respuesta = blocardo[y+h:y, x:x+w]
    _, respuesta = cv2.threshold(respuesta, 128, 255, cv2.THRESH_BINARY)
    
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
    """ 
    Recibe una imagen, devuelve un diccionario con los caracteres, espacios y cantidad de palabras
    
    Esta funcion es la encargada de detectar cuantos caracteres encuentra en cada linea del encabezado de la imagen y de determinar cuando 
    hay un espacio entre ellas y asi determinar la cantidad de palabras que posee.
    """
    
    imagen_encabezado = cv2.adaptiveThreshold(imagen_encabezado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, hierarchy = cv2.findContours(imagen_encabezado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    umbral_area = 5  # Definimos el umbral del área que necesitamos
    contornos_filtrados = [cnt for cnt in contours if cv2.contourArea(cnt) > umbral_area]
    umbral_distancia = 15
    espacios = 0

    # Calculamos la distancia entre los contornos filtrados
    for i in range(-1, len(contornos_filtrados) - 1):
        
        x_actual, _, w_actual, _ = cv2.boundingRect(contornos_filtrados[i])
        x_siguiente, _, w_siguiente, _ = cv2.boundingRect(contornos_filtrados[i + 1])

        # Calculamos la distancia entre los componentes de esta forma ya que toma los contornos del ultimo al primero
        distancia = x_actual - x_siguiente

        # Si la distancia, consideramos que hay un espacio entre palabras
        if distancia >= umbral_distancia:
            espacios += 1


    # Dibujar rectángulos alrededor de los contornos filtrados
    # for cnt in contornos_filtrados:
    #     x, y, w, h = cv2.boundingRect(cnt)  # Solo desempaquetar 4 valores
    #     area = cv2.contourArea(cnt)  # Calcular el área por separado
    #     cv2.rectangle(imagen_encabezado, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar la imagen con los rectángulos dibujados
    # plt.imshow(imagen_encabezado, cmap='gray')
    # plt.show()

    # Devolver el número de caracteres, espacios y palabras detectadas
    salida = {"Caracteres": len(contornos_filtrados), "Espacios": espacios, "Palabras": espacios + 1}
    return salida

def generar_imagen_resultados(condiciones, nombres):
    """     
    Esta funcion es la encargada de generar una imagen con los resultados de los examenes, recibe una lista con las condiciones de los examenes
    y una lista con los nombres de los alumnos y genera una imagen con los resultados de los examenes
    """
    
    # Parámetros del texto de para los resultados
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.7
    grosor = 2
    espacio_entre_lineas = 10  # Espacio entre cada nombre
    
    ancho_max_nombres = max(nombre.shape[1] for nombre in nombres)
    alto_total = sum(nombre.shape[0] + espacio_entre_lineas for nombre in nombres)
    
    ancho_total = ancho_max_nombres + 250  # Espacio extra para el texto
    alto_total += 100  # Margen inferior
    
    img_resultado = np.ones((alto_total, ancho_total, 3), dtype=np.uint8) * 255  # Imagen blanca

    eje_y = 50  
    
    for i, nombre_img in enumerate(nombres):
        alto_nombre, ancho_nombre = nombre_img.shape[:2]   # Seleccionamos solo el alto y ancho de la imagen (Obiando los canales)
        img_resultado[eje_y:eje_y+alto_nombre, 50: 50+ancho_nombre] = cv2.cvtColor(nombre_img, cv2.COLOR_GRAY2BGR) # En el eje x definimos 50 de margen izquierde
        
        # Agregamos el texto
        condicion = condiciones[i]
        texto = f"{condicion}"
        

        if condicion == "Aprobado":
            color_texto = (0, 255, 0)
        else:
            color_texto = (0, 0, 255)

        #Definimos las dimensiones del texto en base a parametros de los nombres y una suma de pixeles predefinida
        cv2.putText(img_resultado, texto, (ancho_nombre + 70, eje_y + int(alto_nombre / 2) + 10), fuente, escala, color_texto, grosor)

        # Desplazar eje y para la siguiente linea
        eje_y += alto_nombre + espacio_entre_lineas

    cv2.imwrite("resultados_examenes.png", img_resultado)
    
    plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
    plt.axis('off') 
    plt.show()

nombres = []
condiciones = []
examenes = ['TP1\examen_1.png','TP1\examen_2.png','TP1\examen_3.png','TP1\examen_4.png', 'TP1\examen_5.png']
for examen in examenes:
    
    rta_examen = []
    img = cv2.imread(examen, 2)
    # Detectar líneas verticales y horizontales
    lineas_v = detectar_lineas_verticales(img, 0.6)
    lineas_h = detectar_lineas_horizontales(img, 0.5)
    blocks = division_bloques(lineas_v, lineas_h, img)
    
    # Corto cada campo
    nombre = img[0:30, 60:255]
    fecha = img[0:30, 300:370]
    clase = img[0:30, 430:490]

    
    nombres.append(nombre)

    d_nombre = detectar_caracteres_encabezado(nombre)
    d_fecha = detectar_caracteres_encabezado(fecha)
    d_class = detectar_caracteres_encabezado(clase)
    
    # Nombre
    if d_nombre["Caracteres"] > 25 or d_nombre["Caracteres"] == 0 or  d_nombre["Palabras"] < 2:
        print("Nombre: Mal")
    else:
        print("Nombre: Ok")

    # Fecha
    if d_fecha["Caracteres"] !=8:
        print("Fecha: Mal")
    else:
        print("Fecha: Ok")

    # Clase
    if d_class["Caracteres"] == 1:
        print("Clase: Ok")
    else:
        print("Clase: Mal")
    
    for block in blocks:
        linea_pregunta = detectar_linea_pregunta(block)
        detectar_respuesta(block, linea_pregunta)
    respuestas_correctas = ['C', 'B', 'A', 'D', 'B', 'B', 'A', 'B', 'D', 'D']
    nuevo_orden = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
    
    
    # Reorganizamos la lista de las respuestas de cada examen
    respuestas_correctas_ordenadas = [rta_examen[i] for i in nuevo_orden]
    contador = 0
    
    for i in range(len(respuestas_correctas)):
        if respuestas_correctas[i] == respuestas_correctas_ordenadas[i]:
            print(f'Pregunta {(i+1)}:', ' OK')
            contador += 1
        else:
            print(f'Pregunta {(i+1)}:', ' MAL')
    if contador >=6:
        condiciones.append('Aprobado')
    else:
        condiciones.append('Reprobado')
    print('Puntaje: ',contador,'/10')

generar_imagen_resultados(condiciones, nombres)
