import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def mostrar_imagen(imagen, titulo):
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis('off')
    plt.show()

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

def procesar_video(video_path):
    """Procesa el video y llama a la función de análisis de fotogramas."""
    cap = cv2.VideoCapture(video_path)  # Abre el archivo de video.
    global width, height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video.

    quiet_frame_number = analizar_frames(cap)
    mostrar_imagen(cv2.imread(os.path.join("TP3/frames", f"frame_{quiet_frame_number}.jpg")), 'Fotograma pausa')

    # Imprimir el número del fotograma donde se detectó la quietud
    if quiet_frame_number != -1:
        print(f'El dado se detuvo en el fotograma: {quiet_frame_number}')
    else:
        print('No se detectó quietud en los dados.')

    cap.release()  # Libera el objeto 'cap', cerrando el archivo.
    cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas.

os.makedirs("TP3/frames", exist_ok=True)
procesar_video('TP3/videos/tirada_1.mp4')  # Procesa el video especificado.
