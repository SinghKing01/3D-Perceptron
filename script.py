# ##############################################################################
# Este script de Python detecta las líneas verticales de una puerta dentro de u-
# na imagen dada, y genera las imágenes de salida dibujando las lineas sobre la
# imagen.
# ##############################################################################

from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import math
from skimage.color import rgb2hsv
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
from skimage.color import rgb2gray
from skimage import img_as_ubyte


def clear_cells(H_acum, max_i, max_j, rows, cols, N):
    # esta función limplia las N celdas vecinas de una posición dada en la matriz
    # tenemos una serie de condiciones para detectar los bordes de la matriz ya
    # que si queremos, por ejemplo limpiar las celdas vecinas de la posición (0,0)
    # no existe una fila superior ni una columna anterior
    i_values = np.arange(max_i - N, max_i + N + 1, 1)
    j_values = np.arange(max_j - N, max_j + N + 1, 1)

    for i in i_values:
        for j in j_values:
            #            if (i >= 0 and i <= rows):
            #                if(j >= 0 and j <= cols):
            #                    if(i != max_i and j != max_j):
            #                        H_acum[i][j] = 0
            if (
                i >= 0
                and j >= 0
                and i <= rows
                and j <= cols
                and (i != max_i or j != max_j)
            ):
                H_acum[i][j] = 0


def hough(img):
    # Recibe una imagen y devuelve la matriz de acumulación de Hough

    # obtenemos las dimensiones de la imagen
    height, width = img.shape

    # el máxmio valor de ro es la hipotenusa que tiene la imagen
    ro_max = int(math.sqrt(height ** 2 + width ** 2))
    # la precisión del incremento del ro es de 1
    ro_increment = 1
    # generamos un array desde 0 hasta ro_max con la precisión indicada
    ro_vals = np.arange(0, ro_max, ro_increment)

    # como dice el enunciado, theta_min y theta_max es -10,10 respectivamente
    # la precisión del incremento de Theta es de 1
    theta_min, theta_max, theta_increment = -10, 10, 1
    # generamos un array desde theta_min hasta theta_max con la precisión indicada
    theta_vals_grados = np.arange(theta_min, theta_max, theta_increment)

    # generamos un array H_acum inicializándolo con ceros y con dimensiones de
    # thetas y ros
    H_acum = np.zeros((len(ro_vals) + 1, len(theta_vals_grados) + 1), dtype=np.uint8)

    # detectamos en la imagen los píxeles de color blanco
    edges = np.where(img == 255)
    # creamos una lista de tuplas (x,y) donde estan los puntos blancos
    x_y_list = list(zip(edges[1], edges[0]))

    # para cada tupla en x_y_list
    for tupla in x_y_list:
        # para cada valor de theta: -10, -9, ..., 0, ..., 10
        for theta in theta_vals_grados:
            # calculamos el valor de ro
            ro = tupla[0] * math.cos(theta) + tupla[1] * math.sin(theta)
            ro = int(round(ro))

            # calculamos los valres de s y t
            s = int(round(theta / theta_increment))
            # s + 10: para Theta = -10 -> s = 0 y Theta = 10 -> s = 21
            s = s + 10
            t = int(round(ro / ro_increment))

            # incrementamos los valores en la matriz H_acum
            H_acum[t, s] = H_acum[t, s] + 1
    return H_acum


def get_x_intersections(N, img):
    # N = número de líneas verticales que se quieren detectar

    # H_acum es la matriz de acumulación de Hough
    H_acum = hough(img)
    # fila_central es la linea horizontal justo en el medio de la imagen
    fila_central = int(round(img.shape[0] / 2))

    # número de filas y columnas en la matriz de acumulación de Hough
    rows, cols = H_acum.shape

    # puntos de la eje X donde se cruzan las líneas de la puerta y la fila_central
    verticals = []
    # valor de la matiz de Hough en dicho punto
    max_vals = []
    # coordenadas i,j de la matriz de Hough con los valres máximos
    max_is = []
    max_js = []

    # inicialmente un valor muy grande (infinito)
    init_max_val = math.inf
    # para el número de líneas verticales que se quieren detectar
    for iteracion in range(N):
        # para cada fila de la matriz de Hough
        max_val, max_i, max_j = 0, 0, 0
        for i in range(rows):
            # para cada columna de la matriz
            for j in range(cols):
                # si el el valor actual es mayor al máximo y menor al máximo
                # valor inicial, actualizamos el max_val y las respectivas
                # coordenadas
                if H_acum[i][j] > max_val and H_acum[i][j] < init_max_val:
                    max_val = H_acum[i][j]
                    max_i = i
                    max_j = j
        # guardamos el valor máximo obtenido en la iteración
        # también guardamos las posiciones (i,j) del valor máximo
        max_vals.append(max_val)
        max_is.append(max_i)
        max_js.append(max_j)
        # acutalizamos init_max_val por el nuevo máximo obtenido
        init_max_val = max_val

    # Ahora para los N máximos obtenidos, limpiamos las celdas vecinas
    counter = len(max_vals)
    idx = 0
    while counter > 0:
        if H_acum[max_is[idx]][max_js[idx]] != 0:
            # limpia 4 celdas vecinas
            clear_cells(H_acum, max_is[idx], max_js[idx], rows, cols, 4)
            #            print("cleared {},{}".format(max_is[idx],max_js[idx]))
            idx = idx + 1
        else:
            # esta celda ha sido limpiada por otra celda, entonces hay que
            # borrar de los máximos
            #            print("deleting {},{}".format(max_is[idx],max_js[idx]))
            del max_is[idx]
            del max_js[idx]
            del max_vals[idx]
        counter = counter - 1

    # Para cada máximo final, se obtendrá su recta a partir de los valores de
    # Theta y Ro
    for idx in range(len(max_vals)):
        max_j = max_js[idx] - 10
        max_i = max_is[idx]
        if max_j == 0:
            max_j = 0.01
        intersection_x = -(fila_central * math.sin(max_j) - max_i) / math.cos(max_j)
        # guardamos el punto de la eje X donde se interseccionan
        verticals.append(intersection_x)

    return verticals, max_is, max_js, H_acum


# Esta función clasifica el segmento por puerta, pared o indeterminado, dada una imagen
def classifica_clase(img, seg_init, seg_max):
    # El número de píxeles con el que se trabajará
    px = 30
    # Altura máxima de la foto menos los restantes por la división de los px
    height_max = img.shape[0] - img.shape[0] % px - px

    # contador para contar los fragmentos clasificados como pared o puerta
    count_doors, count_walls = 0, 0
    c_init = seg_init
    # Mientras no llegamos hasta en fin del segmento (menos los px restantes)
    while c_init + px <= seg_max:
        f_init = 0
        # iniciamos desde fila 0
        # Mientras no llegamos hasta el fin de la altura del segmento
        while f_init + px < img.shape[0]:
            # obtenemos el fragmento actual
            fragment_actual = img[f_init : f_init + px, c_init : c_init + px]
            # pasamos la imagen del fragmento de rgb a hsv
            hsv_fragment = rgb2hsv(fragment_actual)
            # para saturación obtenemos los datos del canal 1
            saturacio = hsv_fragment[:, :, 1]
            # para saturación obtenemos los datos del canal 2
            intensitat = hsv_fragment[:, :, 2]

            # calculamos la media de la saturación e intensidad
            mean_saturacio = np.mean(saturacio)
            mean_intensitat = np.mean(intensitat)

            # Para calcular la energia necesitamos pasar la imagen a escala de grises
            gray_fragment = rgb2gray(fragment_actual)
            gray_fragment = img_as_ubyte(gray_fragment)

            # Obtenemos el glcm para luego obtener la energia
            glcm = graycomatrix(gray_fragment, [1], [0], 256, normed=True)
            energy = graycoprops(glcm, "energy")

            # definimos que el plano indicando que x0,x1,x2 corresponden a los 3 valores obtenidos anteriormente
            x0 = mean_saturacio
            x1 = mean_intensitat
            x2 = energy
            # Vector normal del plano: W = [w0,w1,w2,w3], obtenido a partir del entreno del perceptron
            w0 = 38.815
            w1 = -17.061
            w2 = -50.880
            w3 = 6.626

            # calculamos la salida del perceptron con los valores del nuevo fragmento
            p_output = x0 * w0 + x1 * w1 + x2 * w2 + w3
            if p_output[0][0] > 0:
                # si es mayor a 0, es puerta
                count_doors = count_doors + 1
            else:
                # sino es una pared
                count_walls = count_walls + 1
            # pasamos a la siguiente fila
            f_init = f_init + px

        # pasamos a la siguiente columna
        c_init = c_init + px

    #    print("{} -> {}: {}, {}".format(seg_init,seg_max,count_doors,count_walls))
    # clasificamos el segmento según las indicaciones del enunciado
    suma_total = count_doors + count_walls
    ret = None
    if count_doors / suma_total > 0.7:
        ret = 1
    elif count_walls / suma_total > 0.6:
        ret = 2
    else:
        ret = 0

    # devolvemos el valor (0, 1 o 2) según la clasificación
    return ret


# Esta función edita la imagen original con una clasificación dada de los segmentos
def edit_save_img(img, verticals, classificacio, output_name):
    # creamos una copia local de la imagen
    to_save = img
    # Para cada segmento
    for i in range(len(verticals)):
        # obtenemos el inicio y el fin del segmento
        seg_init, seg_max = None, None
        seg_init = verticals[i]
        if i != len(verticals) - 1:
            seg_max = verticals[i + 1]
        elif i == len(verticals) - 1:
            seg_max = img.shape[1]

        #    print("{} -> {}: {}".format(seg_init,seg_max,classificacio[i]))

        # cambiamos los valores de todos los píxeles de segmento en cuestión al color correspondiente
        if classificacio[i] == 0:
            #    negre
            to_save[:, seg_init:seg_max] = 0
        elif classificacio[i] == 2:
            #    blanc
            to_save[:, seg_init:seg_max] = 255

    # Exportamos la imagen con el nombre dado
    io.imsave(output_name, to_save)


# Esta funcion classifica la puerta de las paredes en una imagen, dado los segmentos
def classifica_segments(img, segments, output_name):
    # Obtenemos las dimensiones de la imagen
    img_height = img.shape[0]
    img_width = img.shape[1]

    # Ordenamos los segmentos de menor a mayor
    segments.sort()

    # Solamente guardamos los segmentos que están dentro de la anchura de la imagen
    verticals_ = []
    for i in range(len(segments)):
        if segments[i] >= 0 and segments[i] <= img_width:
            verticals_.append(segments[i])

    verticals = []
    for i in range(len(verticals_)):
        if i != len(verticals_) - 1 and (verticals_[i + 1] > verticals_[i] + 15):
            verticals.append(int(verticals_[i]))
        elif i == len(verticals_) - 1:
            verticals.append(int(verticals_[i]))

    # Añadimos primer segmento es en la posición 0 de la eje X
    verticals.insert(0, 0)

    # El número de píxeles con el cual se trabajará
    px = 30
    # Los píxeles restantes para completar los 30, no los contamos
    # Hacemos un casting de float a int para poder trabajar con la matriz
    round_verticals = []
    for i in range(len(verticals)):
        round_verticals.append(int(verticals[i] - (verticals[i] % px)))

    # empieza el priceso de clasificación
    classificacio = []
    for i in range(len(round_verticals)):
        # indicamos el inicio del segmento y el final del segmento
        seg_init = round_verticals[i]
        if i == len(round_verticals) - 1:
            seg_max = img_width
        else:
            seg_max = round_verticals[i + 1]

        # obtenemos la clase del segmento con la siguiente función
        # 0 = indeterminado, 1 = puerta, 2 = pared
        clase = classifica_clase(img, seg_init, seg_max)
        if clase == 2:
            #    print("{} -> {}: pared".format(seg_init,seg_max))
            classificacio.append(2)
        elif clase == 1:
            #    print("{} -> {}: puerta".format(seg_init,seg_max))
            classificacio.append(1)
        elif clase == 0:
            #    print("{} -> {}: indeterminado".format(seg_init,seg_max))
            classificacio.append(0)
    # Editamos la imagen a partir de la clasificación anterior y guardamos con el nombre dado
    edit_save_img(img, verticals, classificacio, output_name)


original_images = []


def main():

    # indicamos el tamaño de la figura que luego será exportada como una imagen
    fig = plt.figure(figsize=(15, 10))

    # el número de la imágenes que vamos a leer
    num_imgs = 5

    global original_images
    original_images = []
    for i in range(num_imgs):
        img_name = "image{}.JPG".format(i + 1)
        original_images.append(io.imread(img_name))

    # array de las imágenes leídas
    img_arr = []
    for i in range(num_imgs):
        img_name = "img{}_canny.JPG".format(i + 1)
        img_arr.append(io.imread(img_name))

    counter = 1
    # para cada imágen en el array
    for img in img_arr:
        # llamamos a la función get_x_intersections indicando N y img, donde N
        # es el número de líneas que queremos dibujar. La función nos devuelve
        # en la variable verticals los puntos de la eje X dónde se tienen que
        # dibujar las líneas verticals
        verticals, max_is, max_js, H_acum = get_x_intersections(6, img)

        # para cada punto del eje X en varticals
        for xs in verticals:
            # si el punto es mayor a 0 y menor a la anchura máxima de la imagen
            if xs > 0 and xs < img.shape[1]:
                # dibujamos la linea sobre la imagen

                plt.axvline(x=xs)

        io.imshow(img)
        # exportamos la imagen en la carpeta de trabajo
        plt.savefig("img{}_output.PNG".format(counter))

        # hacemos un clear del plot para que no dibuje las mismas líneas en las
        # imágenes posteriores
        plt.clf()

        # classificamos la puerta de las paredes con la siguiente función
        # pasamos la imagen original, los puntos de la eje X donde se encuentran los verticales
        # y pasamos el nombre con el que se quiere guardar la imagen final classificado
        classifica_segments(
            original_images[counter - 1],
            verticals,
            "img{}_classificat.PNG".format(counter),
        )
        counter = counter + 1

    print("Images generated!")


if __name__ == "__main__":
    main()
