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

def load_imgs(folder_dir,target):
    ret = []
    images = Path(folder_dir).glob('*.JPG')
    for image in images:
        patch = io.imread(image)
        hsv_fragment = rgb2hsv(patch)
        saturacio = hsv_fragment[:, :, 1]
        intensitat = hsv_fragment[:, :, 2]
                    
        mean_saturacio = np.mean(saturacio)
        mean_intensitat = np.mean(intensitat)
                    
        gray_fragment = rgb2gray(patch)
        gray_fragment = img_as_ubyte(gray_fragment)                
        glcm = graycomatrix(gray_fragment,[1],[0],256,normed=True)
        energy = graycoprops(glcm,'energy')
        energy = energy[0][0]
        
        W = [mean_saturacio,mean_intensitat,energy,target]
        ret.append(W)
    
    return ret
    
def get_rand_w(n):
    ret = []
    for i in range(n):
        ret.append(random.uniform(0, 1));
    return ret
        

def get_plane():
    doors_data = load_imgs('patches/door',1)
    walls_data = load_imgs('patches/wall',2)
    
    data = np.concatenate((doors_data, walls_data), axis=0)
    height, width = np.shape(data)
    
#    w1,w2,w3,w0 = 0.0975,0.9952,0.3467,0.6324
#    w = np.array([w1,w2,w3,w0])
    w = get_rand_w(width)
    
    x = np.delete(data,[width-1],axis=1)
    ones = np.ones((len(data),1))
    x = np.hstack((x,ones))
    
    rho = 0.1
    nit = 2000
    
    best_ic = math.inf
    best_w = []
    for t in range(nit):
        suma = np.zeros(width)
        ic = 0
        for k in range(len(data)):
            xi = x[k]
            suma_producto = 0
            for j in range(len(xi)):
                suma_producto = suma_producto + xi[j]*w[j]
            if suma_producto < 0 and data[k][width-1] == 1:
                suma = suma + rho*xi
                ic = ic + 1
            elif suma_producto > 0 and data[k][width-1] == 2:
                suma = suma -rho*xi
                ic = ic + 1
        w = w + suma
        if ic < best_ic:
#            print("{}->{}".format(best_ic,ic))
            best_ic = ic
            best_w = w
        if ic == 0:
            break
    
#    print(w)
    print("best W = {}".format(best_w))
    print("best ic = {}".format(best_ic))
    return best_w, data

def main():
    normal,data = get_plane()
    col_mean = np.mean(data,axis=0)
    x_mean = col_mean[0]
    y_mean = col_mean[1]
    z_mean = col_mean[2]
    
    point = np.array([x_mean,y_mean,z_mean])

    d = -point.dot(normal[0:(np.shape(data)[1]-1)])
    xx, yy = np.meshgrid(range(2), range(2))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    
    fig = plt.figure(figsize = (20, 15))
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, z, alpha=0.2)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    for d in data:
        x = d[0]
        y = d[1]
        z = d[2]
        if(d[3] == 1):
            ax.scatter3D(x,y,z, color = "green", s = 11)
        else:
            ax.scatter3D(x,y,z, color = "red", s = 11)

if __name__ == "__main__":
    main()