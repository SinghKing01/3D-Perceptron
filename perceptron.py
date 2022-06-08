import math
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path
from skimage.color import rgb2hsv
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import random


def analyze_plane(W,data):
    print(W)
    print(data)
    ones = np.ones((len(data),1))
    print(ones)
    data = np.hstack(ones,data)
    print(data)
#    w0,w1,w2,w3 = W
#    miss_classifieds = []
#    well_classifieds = []
#    miss_classifieds_factor = []
#    
#    for d in data:
#        x0 = 1
#        x1 = d[0]
#        x2 = d[1]
#        x3 = d[2]
#        
#        output = w0*x0 + w1*x1 + w2*x2 + w3*x3
#                
#        if(output < 0 and d[3] == DOOR):
#            miss_classifieds.append([x0,x1,x2,x3])
#            miss_classifieds_factor.append(1)
#        elif(output > 0 and d[3] == WALL):
#            miss_classifieds.append([x0,x1,x2,x3])
#            miss_classifieds_factor.append(-1)
#        else:
#            well_classifieds.append([x0,x1,x2,x3])
#        
#        break
#    print([w0,w1,w2,w3])
#    print(miss_classifieds)
#    return miss_classifieds, miss_classifieds_factor, well_classifieds
        
    

#supondré que para door el output es 1 y para pared es 0
DOOR = 1
WALL = 0

doors_data = []
folder_dir = 'patches/door'
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
    
    data = [mean_saturacio,mean_intensitat,energy,DOOR]
    doors_data.append(data)
    
walls_data = []
folder_dir = 'patches/wall'
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
    
    data = [mean_saturacio,mean_intensitat,energy,WALL]
    walls_data.append(data)

data = np.concatenate((doors_data, walls_data), axis=0)
col_mean = np.mean(data,axis=0)
x_mean = col_mean[0]
y_mean = col_mean[1]
z_mean = col_mean[2]
point  = np.array([x_mean, y_mean, z_mean])

fig = plt.figure(figsize = (20, 15))
ax = plt.axes(projection='3d')

w0 = 0
#w1 = random.uniform(0, 1)
#w2 = random.uniform(0, 1)
#w3 = random.uniform(0, 1)
w1 = 0.1
w2 = 0.2
w3 = 0.3
W = [w0,w1,w2,w3]

learning_rate = 0.7
iterations = 3

w0_prima = None
w1_prima = None
w2_prima = None
w3_prima = None

miss_class, miss_class_factor, well_class = analyze_plane([w0,w1,w2,w3],data[0:5])
for i in range(iterations):
    for j in range(len(miss_class)):
        if j == len(miss_class):
            break
        X = miss_class[j]
        factor = miss_class_factor[j]

        w0_prima = w0 + learning_rate*(factor)*X[0]
        w1_prima = w1 + learning_rate*(factor)*X[1]
        w2_prima = w2 + learning_rate*(factor)*X[2]
        w3_prima = w3 + learning_rate*(factor)*X[3]
    
        miss_class_prima, miss_class_factor_prima, well_class_prima = analyze_plane([w0_prima,w1_prima,w2_prima,w3_prima],data)
    
        if(len(miss_class) > len(miss_class_prima)):
            miss_class = miss_class_prima
            miss_class_factor = miss_class_factor_prima
            well_class = well_class_prima
        
            w0 = w0_prima
            w1 = w1_prima
            w2 = w2_prima
            w3 = w3_prima

normal = np.array([w1, w2, w3])
d = -point.dot(normal)
xx, yy = np.meshgrid(range(2), range(2))
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
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