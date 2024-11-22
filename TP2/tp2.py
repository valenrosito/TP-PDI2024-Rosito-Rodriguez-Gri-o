import matplotlib.pyplot as plt
import numpy as np
import cv2

# Cargamos la imagen y le aplicamos un filtro de desenfoque para eliminar el ruido
image = cv2.imread('TP2/monedas.jpg')
image = cv2.blur(image, (7, 7))

# Convertimos la imagen a hls y aplicamos sobre el canal de saturacion un filtro de desenfoque 
hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
h, l, s = cv2.split(hls)
image_blureada = cv2.blur(s, (3, 3))

# Aplicamos un umbral para obtener una imagen binaria y poder realizar operaciones morfologicas
_, imagen_binaria = cv2.threshold(image_blureada, 12, 255, cv2.THRESH_BINARY)


matriz = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
clausura = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, matriz)
apertura = cv2.morphologyEx(clausura, cv2.MORPH_OPEN, matriz)

matriz = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
erode = cv2.erode(apertura, matriz)


contornos, _ = cv2.findContours(erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
mascara = np.zeros_like(erode, dtype=np.uint8)

monedas = []
dados = []
for contorno in contornos:
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    fp = area / (perimetro**2)
    if fp > 0.06:
        monedas.append((contorno, area))
    else:
        dados.append((contorno, area))

# Ordenamos la lista de monedas en base a su area para facilitar la visualziacion
sorted_monedas = sorted(monedas , key=lambda x: x[1])
ten_cents = 0
fifty_cents = 0
one_peso = 0

for moneda in sorted_monedas:
    if moneda[1] > 95000:
        fifty_cents += 1
    elif moneda[1] < 65000:
        ten_cents += 1
    elif moneda[1] < 85000 and moneda[1] > 65000:
        one_peso += 1

print(f"Se encontraron las siguientes cantidades de monedas:\nMonedas 10 centavos: {ten_cents}\nMonedas 50 centavos: {fifty_cents}\nMonedas 1 peso: {one_peso}")


