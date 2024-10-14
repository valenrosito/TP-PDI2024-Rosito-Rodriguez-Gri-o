import cv2
import numpy as np
from matplotlib import pyplot as plt

def local_histogram(img, M, N):
    # Extendemos los bordes de la imagen
    img_expand = cv2.copyMakeBorder(img, M//2, M//2, N//2, N//2, cv2.BORDER_REPLICATE)

    # Imagen de salida
    img_equalized = np.zeros_like(img)

    # Deslizamos el kernel por la imagen
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Definimos el kernel en (i, j)
            window = img_expand[i:i+M, j:j+N]
            # Ecualizacion del histograma local
            img_equalized[i, j] = cv2.equalizeHist(window)[M//2, N//2]

    return img_equalized

img = cv2.imread("TP1/Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)
M, N = 15, 15  # Tamaño del kernel

# Aplicamos la ecualización local del histograma
img_equalized = local_histogram(img, M, N)

# Mostramos la imagen original y la ecualizada
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Imagen Original')
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Imagen Ecualizada (Kernel 18x18)')
plt.imshow(img_equalized, cmap='gray')
plt.show()