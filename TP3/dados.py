import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('TP3')
from utils import *

def analizar_frames(cap):
    """Analiza los fotogramas del video para detectar quietud en los dados."""
    prev_frame = None
    quiet_frames_count = 0
    quiet_threshold = 3  # Número de fotogramas para considerar que los dados están quietos
    frame_count = 0  # Contador de fotogramas
    quiet_frame_number = -1  # Número del fotograma donde se detecta la quietud
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Eliminar 200 píxeles del eje y
        frame = cv2.resize(frame, (int(width / 3), int(height / 3)))
        frame = frame[:- 200, :]

        if frame_count >= 10:
            if prev_frame is not None:
            # Calcular la diferencia entre el fotograma actual y el anterior
                frame_diff = cv2.absdiff(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                _, thresh_diff = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                cv2.imwrite(os.path.join("TP3/frames", f"frame_{frame_count}.jpg"), frame)

                # Contar los píxeles diferentes
                non_zero_count = np.count_nonzero(thresh_diff)

                # Imprimir el número de píxeles diferentes para depuración
                print(f'Fotograma {frame_count}: Píxeles diferentes: {non_zero_count}')

                if non_zero_count < 25:  # Ajustar este valor según sea necesario
                    quiet_frames_count += 1
                else:
                    quiet_frames_count = 0  # Reiniciar el contador si hay movimiento

                # Si los dados están quietos durante el umbral definido, guardar el número del fotograma
                if quiet_frames_count >= quiet_threshold:
                    if quiet_frame_number == -1:  # Solo guardar el primer fotograma en que se detecta la quietud
                        quiet_frame_number = frame_count

            # Mostrar la imagen de diferencia para depuración
            cv2.imshow('Diferencia', thresh_diff)

        # Actualizar el fotograma anterior
        prev_frame = frame.copy()
        frame_count += 1  # Incrementar el contador de fotogramas


        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    return quiet_frame_number



def detectar_dados_con_centroides(frame):
    """Detecta dados en el fotograma, devuelve bounding boxes, máscara y centroides."""
    frame = frame[:-600, :]  # Recortar la parte inferior del frame

    image = cv2.blur(frame, (7, 7))

    # Convertimos la imagen a HLS y aplicamos un desenfoque en el canal de luminancia
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    image_blureada = cv2.blur(h, (3, 3))

    # Umbral binario para obtener una máscara inicial
    _, imagen_binaria = cv2.threshold(image_blureada, 12, 255, cv2.THRESH_BINARY_INV)

    # Operación morfológica para limpiar la máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    open_ = cv2.morphologyEx(imagen_binaria, cv2.MORPH_DILATE, kernel)

    # Detectar componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(open_, connectivity=4)

    dados = []  # Lista de bounding boxes
    centroides = []  # Lista de centroides

    for i in range(1, num_labels):  # Ignorar el fondo (label 0)
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # Filtrar por tamaño del bounding box
        if 40 < h < 160 and 40 < w < 160:  # Ajusta estos valores según las dimensiones de los dados
            dados.append((x, y, w, h))
            centroides.append((int(cx), int(cy)))  # Convertir centroides a enteros

    return dados, open_, centroides


def dibujar_bounding_boxes_y_centroides(frame, bounding_boxes, centroides):
    """Dibuja bounding boxes y centroides en el fotograma."""
    for (x, y, w, h), (cx, cy) in zip(bounding_boxes, centroides):
        # Dibujar el rectángulo alrededor del dado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Dibujar el centroide
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        # Mostrar coordenadas del centroide
        texto = f"({cx}, {cy})"
        cv2.putText(frame, texto, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return frame


import cv2

def mostrar_video_con_centroides(video_path):
    """Procesa el video, detecta componentes conectados y muestra bounding boxes y centroides."""
    cap = cv2.VideoCapture(video_path)
    
    # Obtener propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Crear una ventana que se puede redimensionar
    cv2.namedWindow('Dados y Centroides', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bounding_boxes, mascara, centroides = detectar_dados_con_centroides(frame)
        frame_with_boxes_and_centroids = dibujar_bounding_boxes_y_centroides(frame, bounding_boxes, centroides)

        # Redimensionar el fotograma para que se ajuste a la ventana
        frame_with_boxes_and_centroids = cv2.resize(frame_with_boxes_and_centroids, (int(width / 3), int(height / 3)))

        # Mostrar el resultado
        cv2.imshow('Dados y Centroides', frame_with_boxes_and_centroids)

        if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):  # Ajustar el tiempo de espera según el FPS
            break

    cap.release()
    cv2.destroyAllWindows()


# Ejecutar el procesamiento de video
mostrar_video_con_centroides('TP3/videos/tirada_4.mp4')
