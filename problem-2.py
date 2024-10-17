import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('TP1\examen_1.png', cv2.IMREAD_GRAYSCALE)

# Umbralización binaria invertida (Valores > 128, 0 y Valores <= 128, 255) 
_, img_th = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

# Convertimos la imagen binaria a 1 y 0 (Para facilitar la sumas)
img_th_ones = img_th // 255
img_cols = np.sum(img_th_ones, axis=0)
img_fils = np.sum(img_th_ones, axis=1)

# Definimos umbrales para identificar las lineas
th_col = np.max(img_cols) * 0.6  
th_fil = np.max(img_fils) * 0.5   

# Detectar las posiciones de las líneas (Bool)
img_cols_th = img_cols > th_col
imh_fils_th = img_fils > th_fil

# Encontramos coordenadas lineas verticales
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

# Encontramos coordenadas líneas horizontales 
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
        
lineas_h.pop(0)

# Dibujamos las líneas detectadas sobre la imagen original para visualización
img_salida = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for (inicio_col, fin_col) in lineas_v:
    cv2.line(img_salida, (inicio_col, 0), (inicio_col, img.shape[0]), (0, 255, 0), 2)

for (inicio_fil, fin_fil) in lineas_h:
    cv2.line(img_salida, (0, inicio_fil), (img.shape[1], inicio_fil), (255, 0, 0), 2)

plt.imshow(img_salida), plt.title('Líneas detectadas')
plt.show()


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
            x1 = lineas_v[j][0]
            x2 = lineas_v[j+1][0]
            y1 = lineas_h[i][0]
            y2 = lineas_h[i+1][0]
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
    
