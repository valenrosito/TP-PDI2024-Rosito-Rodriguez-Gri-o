import cv2
import numpy as np
from tkinter import Tk, Scale, HORIZONTAL
from tkinter import Label
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

imagen = cv2.imread('TP2/monedas.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicamos desenfoque para reducir el ruido y facilitar la deteccion de bordes con Canny
imagen_suave = cv2.GaussianBlur(imagen, (3, 3), 0)
bordes = cv2.Canny(imagen_suave, 30, 190)

# Aplicamos dilatación para conectar bordes
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
bordes_dilatados = cv2.morphologyEx(bordes, cv2.MORPH_DILATE, kernel, iterations=3)
bordes_dilatados = cv2.morphologyEx(bordes_dilatados, cv2.MORPH_CLOSE, kernel, iterations=2)

# Creamos una máscara usando los bordes detectados y luego la aplicamos en la imagen original
mascara_objetos = cv2.bitwise_not(bordes_dilatados)
resultado = cv2.bitwise_and(imagen, imagen, mask=mascara_objetos)

f = mascara_objetos
fd = cv2.morphologyEx(f, cv2.MORPH_DILATE, kernel)
fo = cv2.morphologyEx(f, cv2.MORPH_OPEN, kernel)
fc = cv2.morphologyEx(f, cv2.MORPH_CLOSE, kernel)
plt.figure()
ax1 = plt.subplot(221); plt.imshow(f, cmap='gray'), plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(fd, cmap='gray'), plt.title('Dilatacion'), plt.xticks([]), plt.yticks([])
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(fo, cmap='gray'), plt.title('Apertura'), plt.xticks([]), plt.yticks([])
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(fc, cmap='gray'), plt.title('Clausura'), plt.xticks([]), plt.yticks([])
plt.show(block=False)