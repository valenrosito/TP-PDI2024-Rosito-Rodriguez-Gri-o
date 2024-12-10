import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append('TP3')

def mostrar_imagen(imagen, titulo):
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis('off')
    plt.show()


def analizar_frames(cap):
    """Analiza los fotogramas del video para detectar quietud en los dados."""
    prev_frame = None
    quiet_frames_count = 0 # Contador de fotogramas quietos
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
def obtener_puntuacion(frame):
    """Calcula la puntuación de los dados en el fotograma dado."""
    # Aquí puedes implementar la lógica para calcular la puntuación de los dados
    # Por ejemplo, podrías usar técnicas de procesamiento de imágenes para detectar los puntos en los dados
    # y sumar la puntuación total.
    # Esta es una implementación de ejemplo y puede necesitar ajustes según tus necesidades.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    puntuacion = 0
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Ajustar este valor según sea necesario
            puntuacion += 1
    return puntuacion

def analizar_frames_y_obtener_puntuacion(cap):
    """Analiza los fotogramas del video para detectar quietud en los dados y obtener la puntuación final."""
    quiet_frame_number = analizar_frames(cap)
    if quiet_frame_number != -1:
        frame = cv2.imread(os.path.join("TP3/frames", f"frame_{quiet_frame_number}.jpg"))
        puntuacion = obtener_puntuacion(frame)
        print(f'Puntuación final: {puntuacion}')
        return quiet_frame_number, puntuacion
    else:
        print('No se detectó quietud en los dados.')
        return -1, 0

def procesar_video(video_path):
    """Procesa el video y llama a la función de análisis de fotogramas."""
    cap = cv2.VideoCapture(video_path)  
    global width, height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video.

    quiet_frame_number = analizar_frames(cap)

    # Imprimir el número del fotograma donde no hubo cambios
    if quiet_frame_number != -1:
        print(f'El dado se detuvo en el fotograma: {quiet_frame_number}')
        mostrar_imagen(cv2.imread(os.path.join("TP3/frames", f"frame_{quiet_frame_number}.jpg")), 'Fotograma pausa')
    else:
        print('No se detectó quietud en los dados.')

    cap.release()  # Libera el objeto 'cap', cerrando el archivo.
    cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas.

def guardar_video(frames, path, fps=30, codec="mp4v"):    
    
    if not frames:
        raise ValueError("No hay frames en la lista.")
    os.makedirs(os.path.dirname(path), exist_ok=True) # Crea el directorio si no existe
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*codec) 
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()

    print(f"El video se guardo en: {path}")

def procesar_videos_en_carpeta(carpeta_path):
    """Itera sobre todos los videos en una carpeta y los procesa."""
    rutas = []
    for archivo in os.listdir(carpeta_path):
        if archivo.endswith('.mp4'):
            video_path = os.path.join(carpeta_path, archivo)
            rutas.append(video_path)
    return rutas
