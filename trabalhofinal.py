import cv2
import numpy as np
import time

# Carrega as imagens
img1 = cv2.imread(r'D:/Python/Nova pasta/WhatsApp Image 2023-03-16 at 21.44.00 (1).jpeg')
img2 = cv2.imread(r'D:/Python/Nova pasta/WhatsApp Image 2023-03-16 at 21.44.00.jpeg')

bilateral_filter_size = 2
bilateral_sigma_space = 75
bilateral_sigma_color = 75

img1 = cv2.bilateralFilter(img1, bilateral_filter_size, bilateral_sigma_color, bilateral_sigma_space)
img2 = cv2.bilateralFilter(img2, bilateral_filter_size, bilateral_sigma_color, bilateral_sigma_space)

# Imagens em escala de cinza
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)



# Definir os parâmetros para o detector de Harris
block_size = 2
ksize = 3
k = 0.01



gray1 = np.float32(gray1)
gray2 = np.float32(gray2)

# Detectar os cantos nas duas imagens usando o detector de Harris
corners1 = cv2.cornerHarris(gray1, block_size, ksize, k)
corners2 = cv2.cornerHarris(gray2, block_size, ksize, k)

# Aplica NMS
nms_size = 5
corners1_max = cv2.dilate(corners1, np.ones((nms_size, nms_size)))
corners1[corners1 < corners1_max] = 0

corners2_max = cv2.dilate(corners2, np.ones((nms_size, nms_size)))
corners2[corners2 < corners2_max] = 0

# Criar um kernel para dilatação
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))

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
