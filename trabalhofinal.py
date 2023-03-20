import cv2
import numpy as np
import time

def loading_bar(current,total):
    toolbar_width = 40
    progress = int(current/total*toolbar_width)
    bar = "[" + " "*(toolbar_width-progress) + "#"*(progress) + "]"
    print(f"\r{bar} {current}/{total}", end="", flush=True)
    time.sleep(0.1)


# Carrega as imagens
img1 = cv2.imread(r'D:/Python/Nova pasta/WhatsApp Image 2023-03-16 at 21.44.00 (1).jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r'D:/Python/Nova pasta/WhatsApp Image 2023-03-16 at 21.44.00.jpeg', cv2.IMREAD_GRAYSCALE)

# Definir os parâmetros para o detector de Harris
block_size = 2
ksize = 3
k = 0.001

# Detectar os cantos nas duas imagens usando o detector de Harris
corners1 = cv2.cornerHarris(img1, block_size, ksize, k)
corners2 = cv2.cornerHarris(img2, block_size, ksize, k)

# Criar um kernel para dilatação
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# Dilatar as regiões brancas (cantos) na imagem
corners1_dilated = cv2.dilate(corners1, kernel)
corners2_dilated = cv2.dilate(corners2, kernel)

img1 = cv2.imread(r'D:/Python/Nova pasta/WhatsApp Image 2023-03-16 at 21.44.00 (1).jpeg')
img2 = cv2.imread(r'D:/Python/Nova pasta/WhatsApp Image 2023-03-16 at 21.44.00.jpeg')

# Pintar os cantos na imagem
img1[corners1_dilated > 0.01*corners1_dilated.max()] = [0, 0, 255]
img2[corners2_dilated > 0.01*corners2_dilated.max()] = [0, 0, 255]

# Mostrar as imagens com os cantos marcados
cv2.imshow("Imagem 1", img1)
cv2.imshow("Imagem 2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
