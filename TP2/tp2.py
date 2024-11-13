import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread('TP2/monedas.jpg')
imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Separamos el canal V para mejorar el contraste
h, s, v = cv2.split(imagen_hsv)

# Desenfocamos para eliminar ruido
v_suavizado = cv2.GaussianBlur(v, (21, 21), 0)
v_suavizado = cv2.add(v_suavizado, 4)

# Restamos el fondo suavizado del canal V original
v_sin_fondo = cv2.subtract(v, v_suavizado)

# Aplicamos ecualización adaptativa en el canal V sin fondo
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
v_ecualizado = clahe.apply(v_sin_fondo)

# Recomponemos la imagen HSV con el canal V ecualizado
imagen_hsv_ecualizada = cv2.merge([h, s, v_ecualizado])
imagen_contraste = cv2.cvtColor(imagen_hsv_ecualizada, cv2.COLOR_HSV2BGR)
imagen_gris = cv2.cvtColor(imagen_contraste, cv2.COLOR_BGR2GRAY)

# Aplicamos desenfoque para reducir el ruido y facilitar la deteccion de bordes con Canny
imagen_suave = cv2.GaussianBlur(imagen_gris, (5, 5), 0)
bordes = cv2.Canny(imagen_suave, 50, 150)

# Aplicamos dilatación para conectar bordes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
bordes_dilatados = cv2.dilate(bordes, kernel, iterations=2)

# Creamos una máscara usando los bordes detectados y luego la aplicamos en la imagen original
mascara_objetos = cv2.bitwise_not(bordes_dilatados)
resultado = cv2.bitwise_and(imagen, imagen, mask=mascara_objetos)

# Aplicamos operaciones morfológicas adicionales para reducir el ruido
mascara_objetos = cv2.morphologyEx(mascara_objetos, cv2.MORPH_CLOSE, kernel)
mascara_objetos = cv2.morphologyEx(mascara_objetos, cv2.MORPH_OPEN, kernel)

f = mascara_objetos
fd = cv2.dilate(f, kernel, iterations=2)
fe = cv2.erode(f, kernel)
fmg = cv2.morphologyEx(f, cv2.MORPH_GRADIENT, kernel)
plt.figure()
ax1 = plt.subplot(221); plt.imshow(f, cmap='gray'), plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(fd, cmap='gray'), plt.title('Dilatacion'), plt.xticks([]), plt.yticks([])
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(fe, cmap='gray'), plt.title('Erosion'), plt.xticks([]), plt.yticks([])
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(fmg, cmap='gray'), plt.title('Gradiente Morfologico'), plt.xticks([]), plt.yticks([])
plt.show(block=False)