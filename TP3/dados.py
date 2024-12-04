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

def detectar_dados(frame):
    """Detecta dados en el fotograma y devuelve las bounding boxes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 120)


    kernel = np.ones((21, 21), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara = np.zeros_like(gray)
    cv2.drawContours(mascara, contours, -1, 255, -1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    bounding_boxes = []

    for i in range(1, num_labels):  # Empezar desde 1 para ignorar el fondo
        x, y, w, h, area = stats[i]
        if 70 <= w <= 100 and 70 <= h <= 100:
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes

def dibujar_bounding_boxes(frame, bounding_boxes):
    """Dibuja los bounding boxes en el fotograma."""
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def mostrar_video_con_bounding_boxes(video_path):
    """Procesa el video y muestra los bounding boxes en cada fotograma."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 

    frame_count = 0

    out = cv2.VideoWriter('Output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bounding_boxes = detectar_dados(frame)
        if frame_count >= 30:
            frame_with_boxes = dibujar_bounding_boxes(frame, bounding_boxes)
            frame_with_boxes = cv2.resize(frame_with_boxes, (int(width / 3), int(height / 3)))
            out.write(frame_with_boxes)
            cv2.imshow('Dados', frame_with_boxes)
            
        else: 
            cv2.imshow('Dados', frame)
            
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        frame_count += 1  # Incrementar el contador de fotogramas

# os.makedirs("TP3/frames", exist_ok=True)
os.makedirs("TP3/videos_outpu", exist_ok=True)
# procesar_video('TP3/videos/tirada_1.mp4')  

mostrar_video_con_bounding_boxes('TP3/videos/tirada_2.mp4')  