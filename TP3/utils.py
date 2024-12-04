import cv2
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('TP3')
from dados import analizar_frames

def mostrar_imagen(imagen, titulo):
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis('off')
    plt.show()


def procesar_video(video_path):
    """Procesa el video y llama a la función de análisis de fotogramas."""
    cap = cv2.VideoCapture(video_path)  
    global width, height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video.

    quiet_frame_number = analizar_frames(cap)
    mostrar_imagen(cv2.imread(os.path.join("TP3/frames", f"frame_{quiet_frame_number}.jpg")), 'Fotograma pausa')

    # Imprimir el número del fotograma donde no hubo cambios
    if quiet_frame_number != -1:
        print(f'El dado se detuvo en el fotograma: {quiet_frame_number}')
        mostrar_imagen(cv2.imread(os.path.join("TP3/frames", f"frame_{quiet_frame_number}.jpg")), 'Fotograma pausa')
    else:
        print('No se detectó quietud en los dados.')

    cap.release()  # Libera el objeto 'cap', cerrando el archivo.
    cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas.