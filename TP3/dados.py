from utils import *


def detectar_dados_con_centroides_y_puntos(frame):
    """Detecta dados en el fotograma, devuelve bounding boxes, máscara, centroides y puntos."""
    frame = frame[:-600, :]  # Recortar la parte inferior del frame

    image = cv2.blur(frame, (7, 7))

    # Convertimos la imagen a HLS y aplicamos un desenfoque en el canal de luminancia
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    image_blureada = cv2.blur(h, (3, 3))

    # Umbral binario para obtener una máscara inicial
    _, imagen_binaria = cv2.threshold(image_blureada, 12, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    open_ = cv2.morphologyEx(imagen_binaria, cv2.MORPH_DILATE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(open_, connectivity=4)

    dados = []  # Lista de bounding boxes
    centroides = []  # Lista de centroides
    puntos_por_dado = []  # Lista de puntos detectados en cada dado

    for i in range(1, num_labels):  # Ignoramos el fondo (label 0)
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # Calcula el perímetro
        contorno = labels == i
        perimeter = cv2.arcLength(cv2.findContours(contorno.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True)

        # Calcula el factor de forma
        if perimeter > 0:  
            fp = area / (perimeter ** 2)
        else:
            fp = 0

        # Filtramos por tamaño del bounding box y factor de forma
        if 40 < h < 160 and 40 < w < 160 and fp < 0.1:  
            dados.append((x, y, w, h))
            centroides.append((int(cx), int(cy)))  # Convierte centroides a enteros

            # Detectamos círculos en el área del dado usando HoughCircles
            dado_region = frame[y:y+h, x:x+w]
            gray_dado = cv2.cvtColor(dado_region, cv2.COLOR_BGR2GRAY)
            gray_dado = cv2.GaussianBlur(gray_dado, (7, 7), 2)
            


            circles = cv2.HoughCircles(
                gray_dado, cv2.HOUGH_GRADIENT, 1, 13, param1=30, param2=15, minRadius=5, maxRadius=15
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                puntos_por_dado.append(circles[0, :])  # Guardamos los círculos detectados
            else:
                puntos_por_dado.append([])  # Si no se detectan círculos, guardamos una lista vacía

    return dados, open_, centroides, puntos_por_dado

def calcular_puntuacion_dados(puntos_por_dado):
    """Calcula la puntuación total de los dados y determina la combinación."""
    combinacion = [len(puntos) for puntos in puntos_por_dado]
    combinacion.sort()
    puntaje = 0
    texto = "No combina, 0 puntos."
    
    if len(combinacion) == 5:
        if len(set(combinacion)) == 1:
            puntaje = 50
            texto = "Generala: 50 puntos"
        elif any(combinacion.count(val) == 4 for val in combinacion):
            puntaje = 40
            texto = "Poker: 40 puntos"
        elif any(combinacion.count(val) == 3 for val in combinacion):
            puntaje = 30
            texto = "Full: 30 puntos"
        elif combinacion == [1, 2, 3, 4, 5]:
            puntaje = 20
            texto = "Escalera menor: 20 puntos"
        elif combinacion == [2, 3, 4, 5, 6]:
            puntaje = 25
            texto = "Escalera mayor: 25 puntos"
        else:
            puntaje = sum(combinacion)
            texto = f"No combina, {puntaje} puntos."
    
    return puntaje, texto

def dibujar_bounding_boxes_centroides_y_puntos(frame, bounding_boxes, centroides, puntos_por_dado):
    """Dibuja bounding boxes, centroides, puntos en el fotograma y muestra la puntuación sobre las boxes."""
    for (x, y, w, h), (cx, cy), puntos in zip(bounding_boxes, centroides, puntos_por_dado):
        # Dibujar el rectángulo alrededor del dado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Dibujar el centroide
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Mostrar coordenadas del centroide
        texto = f"({cx}, {cy})"
        cv2.putText(frame, texto, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Contar los puntos (círculos) detectados dentro del dado
        cantidad_puntos = len(puntos)
        
        # Mostrar la puntuación sobre la bounding box
        cv2.putText(frame, f"Puntuacion: {cantidad_puntos}", (x-50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Dibujar los puntos (círculos) detectados dentro del dado
        for (x_circ, y_circ, r) in puntos:
            cv2.circle(frame, (x + x_circ, y + y_circ), r, (255, 0, 0), 2)  # Círculo azul para los puntos

    # Calcular la puntuación total y el texto de la combinación
    puntaje, texto_combinacion = calcular_puntuacion_dados(puntos_por_dado)
    
    # Mostrar el texto de la combinación en el frame
    cv2.putText(frame, texto_combinacion, (250, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)

    return frame


def mostrar_video_con_centroides_y_puntos(video_path):
    """Procesa el video, detecta componentes conectados, puntos de los dados y muestra bounding boxes, centroides y puntos."""
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []

    cv2.namedWindow('Dados, Centroides y Puntos', cv2.WINDOW_NORMAL)

    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            bounding_boxes, mascara, centroides, puntos_por_dado = detectar_dados_con_centroides_y_puntos(frame)
            frame_with_boxes_centroides_y_puntos = dibujar_bounding_boxes_centroides_y_puntos(
                frame, bounding_boxes, centroides, puntos_por_dado
            )

            # Redimensionamos el fotograma para que se ajuste a la ventana
            frame_with_boxes_centroides_y_puntos = cv2.resize(frame_with_boxes_centroides_y_puntos, (int(width / 3), int(height / 3)))
            frames.append(frame_with_boxes_centroides_y_puntos)

            cv2.imshow('Dados, Centroides y Puntos', frame_with_boxes_centroides_y_puntos)

        key = cv2.waitKey(1000 // fps) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    return frames


routes = procesar_videos_en_carpeta('TP3/videos')
num = 0

for route in routes:
    procesar_video(route)

# Ejecutar el procesamiento de video con detección de puntos
for route in routes:
    num += 1
    guardar_video(mostrar_video_con_centroides_y_puntos(route), 'TP3/videos_output/tirada_' + str(num) + '.mp4')