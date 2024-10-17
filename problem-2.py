import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('examen_1.png', 2)

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

# Detectar líneas verticales y horizontales
lineas_v = detectar_lineas_verticales(img, 0.6)
lineas_h = detectar_lineas_horizontales(img, 0.5)
lineas_h.pop(0)

respuestas = ['C', 'D', 'A', 'D', 'B', 'B', 'A', 'B', 'D', 'D']
blocks = []

# Verificamos que las líneas no estén vacías
if len(lineas_v) == 0 or len(lineas_h) == 0:
    print("Error: No se detectaron líneas horizontales o verticales.")
else:
    print(f'Líneas verticales detectadas: {lineas_v}')
    print(f'Líneas horizontales detectadas: {lineas_h}')

    count = 0
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
            count += 1

    fig, axs = plt.subplots(len(blocks)//2, 2, figsize=(10, 10))
    axs = axs.flatten()
    for i, block in enumerate(blocks):
        axs[i].imshow(block, cmap='gray')
        axs[i].set_title(f'Bloque {i+1}')
        axs[i].axis('off'), axs[i].set_xticks([]), axs[i].set_yticks([])  # Oculta los ejes
    plt.tight_layout()  
    plt.show()

    print(f'Se detectaron {len(blocks)} blocks.')


for block in blocks:
    lineas_horizontales1 = detectar_lineas_horizontales(block, 0.75)
    for inicio, fin in lineas_horizontales1:
        cv2.line(block, (0, inicio), (block.shape[1], inicio), (0, 255, 0), 2)  # Dibuja la línea en verde
    
fig, axs = plt.subplots(len(blocks)//2, 2, figsize=(10, 10))
axs = axs.flatten()
for i, block in enumerate(blocks):
    axs[i].imshow(block, cmap='gray')
    axs[i].set_title(f'Bloque {i+1}')
    axs[i].axis('off'), axs[i].set_xticks([]), axs[i].set_yticks([])  # Oculta los ejes
plt.tight_layout()  
plt.show()


# respuestas_correctas = ['A', 'B', 'C', 'D', 'B', 'A', 'C', 'B', 'D', 'A']
# resultados = []
# resp_corr_cont = 0