
# ##############################################################################
# Este script de Python ha sido utilizado para extraer contornos de las imagenes
# Tenemos que indicar el nombre de la imagen de entrada y de la imagen de salida
# ##############################################################################

from skimage import io
from skimage.color import rgb2gray
from skimage import exposure
from skimage import img_as_ubyte
from skimage.filters.rank import median
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import filters
from skimage import feature

rows = 3
cols = 3
figura = plt.figure(figsize=(15,13))

# Lectura imagen de entrada
img1 = io.imread("image5.JPG")
# Conversion en escala de grises
img1 = rgb2gray(img1)

# Para mostrar diferentes pasos en una misma figura
figura.add_subplot(rows,cols,1)
plt.imshow(img1,norm=colors.NoNorm(),cmap='gray')
plt.title("Original")

# Binarizacion de la imagen
# He utilizado el umbral que nos proporciona Otsu
otsu_img1 = filters.threshold_otsu(img1)
bin_img1 = (img1 > otsu_img1)*255

# Mostramos la imagen binarizada en la misma figura
figura.add_subplot(rows,cols,2)
plt.imshow(bin_img1,norm=colors.NoNorm(),cmap='gray')
plt.title("Binarización Otsu")

# Extracción de contornos a partir de la imagen binarizada
# He utilizado los diferentes métodos para extraer los contornos, sin embargo
# el mejor resultado nos ha dado el filtro de Canny

# img1_sobel = filters.sobel(img1)
# img1_roberts = filters.roberts(img1)
# img1_prewitt = filters.prewitt(img1)
img1_canny = feature.canny(img1)

# figura.add_subplot(rows,cols,3)
# plt.imshow(img1_sobel,norm=colors.NoNorm(),cmap='gray')
# plt.title("SOBEL")

# figura.add_subplot(rows,cols,4)
# plt.imshow(img1_roberts,norm=colors.NoNorm(),cmap='gray')
# plt.title("ROBERTS")

# figura.add_subplot(rows,cols,5)
# plt.imshow(img1_prewitt,norm=colors.NoNorm(),cmap='gray')
# plt.title("PREWITT")

figura.add_subplot(rows,cols,6)
plt.imshow(img1_canny,cmap='gray')
plt.title("CANNY")

io.imshow(img1_canny)

io.imsave("img5_canny.JPG",img1_canny)
# io.imsave("img5_prewitt.JPG",img1_prewitt)
