# ##############################################################################
# Este script de Python implementa perceptron nDimensional, dependiendo de las
# dimensiones de los datos sobre los cuales se trabaja.
# ##############################################################################

from skimage import io
import math
import numpy as np
from pathlib import Path
from skimage.color import rgb2hsv
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random

# esta función devuelve el array de las imagenes guardadas en directorio que se especifica en el parámetro
# además, indicamos a la función el target, es decir, como queremos clasificarlo. Por ejemplo, clase 1 o 2.
def load_imgs(folder_dir, target):
    # este será el array que se devuelve
    ret = []
    # guardamos en un array los nombres de todos los archivos que se acaban en .JPG
    images = Path(folder_dir).glob("*.JPG")
    # para cada nombre del archivo JPG
    for image in images:
        # leemos la imagen
        patch = io.imread(image)
        # pasamos a hsv para obtener la saturación e intensidad
        hsv_fragment = rgb2hsv(patch)

        # obtenemos la saturación e intensidad con los canales 1 y 2 respectivamente
        saturacio = hsv_fragment[:, :, 1]
        intensitat = hsv_fragment[:, :, 2]

        # hacemos la media de los dos arrays anteriores
        mean_saturacio = np.mean(saturacio)
        mean_intensitat = np.mean(intensitat)

        # pasamos el patch origina a escala de grises para obtener la energia
        gray_fragment = rgb2gray(patch)
        gray_fragment = img_as_ubyte(gray_fragment)
        # obtenemos el glcm
        glcm = graycomatrix(gray_fragment, [1], [0], 256, normed=True)
        # obtenemos la energía
        energy = graycoprops(glcm, "energy")
        energy = energy[0][0]

        # creamos el array con los valores anteriores y añadimos al dataset
        X = [mean_saturacio, mean_intensitat, energy, target]
        ret.append(X)

    return ret


# Esta función generta una array con longitud n, con valores random entre 0 y 1
def get_rand_w(n):
    ret = []
    for i in range(n):
        ret.append(random.uniform(0, 1))
    return ret


# Esta función devuelve un plano que separe las dos clases que existen en los datos
def get_plane(data):
    # obtenemos la altura y anchura de los datos (filas y columnas)
    height, width = np.shape(data)

    # w1,w2,w3,w0 = 0.0975,0.9952,0.3467,0.6324
    # w = np.array([w1,w2,w3,w0])

    # Obtenemos un array con n valores random entre 0 y 1
    w = get_rand_w(width)

    # creamos una nueva tabla a partir de los datos con nombre x y eliminamos la columna de clasificación
    x = np.delete(data, [width - 1], axis=1)
    # creamos y añadimos a esta tabla una columna con todos 1's (bias)
    ones = np.ones((len(data), 1))
    x = np.hstack((x, ones))

    # definimos que rho=0.1
    rho = 0.1
    # definimos que el número de iteraciones es 2000
    nit = 2000

    # inicializamos el número de clasificaciones incorrectas (ic) con un número muy grande
    best_ic = math.inf
    # en best_w guardamos el mejor plano, que menos ic tenga
    best_w = []
    # tecorremos un bucle hasta el número de iteraciones
    for t in range(nit):
        # creamos un array iniciacliado con ceros
        suma = np.zeros(width)
        # inicalizamos ic con 0
        ic = 0
        # recorremos todo el dataset
        for k in range(len(data)):
            # obtenemos la fila i, contiene la media de saturación,intensidad,energia y 1
            xi = x[k]
            # en suma producto se guarda de xi * w' + suma_producto
            suma_producto = 0
            for j in range(len(xi)):
                suma_producto = suma_producto + xi[j] * w[j]
            # si la suma es menor a cero (pared), pero el fragmento en realidad era de una puerta
            # entonces lo ha clasificado mal y lo mismo para la otra clasificación.
            # si se ha clasificado mal, se cambia el valor de suma y se incrementa el valor de ic
            if suma_producto < 0 and data[k][width - 1] == 1:
                suma = suma + rho * xi
                ic = ic + 1
            elif suma_producto > 0 and data[k][width - 1] == 2:
                suma = suma - rho * xi
                ic = ic + 1
        # actualizamos el array w
        w = w + suma
        # si el número de ic es menor que antes, actualizamos best_ic con ic y best_w con el nuevo w
        if ic < best_ic:
            #    print("{}->{}".format(best_ic,ic))
            best_ic = ic
            best_w = w
        if ic == 0:
            # si no hay ninguna muestra mal clasificada, paramos el proceso
            break

    # print(w)
    print("best W = {}".format(best_w))
    print("best ic = {}".format(best_ic))
    return best_w


def main():
    # leemos los datos sobre los cuales trabajaremos y especificamos la clase 1 o 2
    doors_data = load_imgs("patches/door", 1)
    walls_data = load_imgs("patches/wall", 2)
    # juntamos los dos arrays anteriores
    data = np.concatenate((doors_data, walls_data), axis=0)
    # la tabla data tiene las siguiente 4 columnas respectivamente:
    # media_saturación,media_intensidad,energía,clasificación = (1 o 2)

    # Obtenemos el plano a partir de la función get_plane
    [w1, w2, w3, w0] = get_plane(data)
    # especificamos el tamaño de la figura del plano
    fig = plt.figure(figsize=(20, 15))
    # creamos un objeto tipo axes para figura en 3d
    ax = plt.axes(projection="3d")

    # Recorremos cada dato (punto 3 dimensional)
    for d in data:
        x = d[0]
        y = d[1]
        z = d[2]
        # según una clase u otra utilzamos color diferente
        if d[3] == 1:
            ax.scatter3D(x, y, z, color="green", s=11)
        else:
            ax.scatter3D(x, y, z, color="red", s=11)

    # crea valor pels eixos x,y
    x = y = np.arange(0.0, 1.0, 0.1)
    [xx, yy] = np.meshgrid(x, y)
    # calcula els valors de z
    z = (-w1 * xx - w2 * yy - w0) / w3
    # pinta el pla de separació
    ax.plot_surface(xx, yy, z, alpha=0.2)


if __name__ == "__main__":
    main()
