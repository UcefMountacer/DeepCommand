# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:14:15 2019

@author: damie
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from glob import glob
import librosa.display
import pickle
import gc
from PIL import Image

directory= "D:/bural/cours/projet/train_img"
dir2="D:/bural/cours/projet/test_img"
EXT = "*.png"

l=[os.path.basename(x[0]) for x in os.walk(directory)]
print(l[1:])
l.pop(0)


""" on met tous les fichiers et le nom de leur dosssier (cat,dog,...) dans une liste data et data_test """

data=[]
for path, subdir, files in os.walk(directory):
    for file in glob(os.path.join(path, EXT)) :
        data.append([file,os.path.basename(path)])

data_test=[]
for path, subdir, files in os.walk(dir2):
    for file in glob(os.path.join(path, EXT)) :
        data_test.append([file,os.path.basename(path)])


""" on creer les dimensions de la zone a isoler sur l'image (on ne veut pas les graduations)"""
    # Define box inside image

left = 54
top = 35
width = 334
height = 216

    # Create Box

box = (left, top, left+width, top+height)


#im=Image.open(data[0][0])
#area=im.crop(box)
#area.show()
#a=np.array(area)

""" on convertit les images en numpy array"""

x_train=[]
i=0
print("data")
for d in data:
    i+=1
    if i%1000==0:
        print(i)
        gc.collect()
    im=Image.open(d[0])
    x_train.append(np.array(im.crop(box)))
y_train=[l.index(d[1]) for d in data]

x_test=[]
i=0
print("data test")
for d in data_test:
    i+=1
    if i%1000==0:
        print(i)
        gc.collect()
    im=Image.open(d[0])
    x_test.append(np.array(im.crop(box)))
y_test=[l.index(d[1]) for d in data_test]


""" convertit les listes en array pour le neural network"""

y_train = np.array(y_train)
x_train = np.stack(x_train)
#x_train = np.vstack(x_train)

x_test = np.stack(x_test)
y_test = np.array(y_test)


""" avec pickle, on sauvegarde les données dans un fichier pour ne pas avoir a retraiter le dataset"""

print("pickling")
f=open( "data_train_image.dtsim", "wb" )
pickle.dump([x_train,y_train,l],f,protocol=4)
f.close()
f=open( "data_test_image.dtsim", "wb" )
pickle.dump([x_test,y_test],f,protocol=4)
f.close()




#im = Image.open("D:/bural/cours/projet/train_img/bed/8aa35b0c_nohash_0.png")
#np_im = np.array(im)
#print(np_im.shape)