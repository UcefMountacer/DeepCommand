# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:14:15 2019

@author: damie
"""

import numpy as np
import os
from glob import glob
import pickle
import gc
from PIL import Image
import random as rd


""" definition des emplacement des dossiers"""

directory= "D:/bural/cours/projet/train_img"
dir2="D:/bural/cours/projet/test_img"
EXT = "*.png"

l=[os.path.basename(x[0]) for x in os.walk(directory)]

l.pop(0)#le premier element est le dossier parent (ici train_img)
#on selectione uniquement certains mots
filtre=["bruit","right","left","on","off","cat","dog","down","up","stop","no","yes","wow","zero","one","two","three","four","five","six","seven","eight","nine","tree","go","house"]
l=filtre[:]


""" on met tous les fichiers et le nom de leur dosssier (cat,dog,...) dans une liste data et data_test """

data=[]
for path, subdir, files in os.walk(directory):
    for file in glob(os.path.join(path, EXT)) :
        if str(os.path.basename(path)) in filtre :
            data.append([file,os.path.basename(path)])
        
rd.shuffle(data)#on melange les donnees pour ne pas avoir 800 fichiers 'cat' suivi de 800 fichiers 'dog' ex....
data_test=[]
for path, subdir, files in os.walk(dir2):
    for file in glob(os.path.join(path, EXT)) :
        if str(os.path.basename(path)) in filtre :
            data_test.append([file,os.path.basename(path)])
rd.shuffle(data_test)


""" on creer les dimensions de la zone a isoler sur l'image (on ne veut pas les graduations)"""

left = 54
top = 35
width = 334
height = 216

box = (left, top, left+width, top+height)

""" on convertit les images en numpy array"""

#size= 67,43
size= 84,54
#size= 167,108
#size= 334,216

x_train=[]
i=0
print("data")
for d in data:
    i+=1
    if i%1000==0:
        """
        tout les 1000 fichiers on affiche i pour avoir une idee de l'avancement du programe
        on appel aussi le garbage collector pour vider en parti la memoire
        """
        print(i)
        gc.collect()
        
    im=Image.open(d[0])#on ouvre l'image
    im=im.crop(box)#on decoupe l'image
    im.thumbnail(size,Image.ANTIALIAS)#on redimensionne l'image (thumbnail permet d'avoir une meilleur qualite mais prend plus de temps)
    x_train.append(np.array(im))#on la converti en array

y_train=[l.index(d[1]) for d in data]

""" on convertit les listes en array pour le neural network """
y_train = np.array(y_train)
x_train = np.stack(x_train)


gc.collect()

x_test=[]
i=0
print("data test")
for d in data_test:
    i+=1
    if i%1000==0:
        print(i)
        gc.collect()
    im=Image.open(d[0])
    im=im.crop(box)
    im.thumbnail(size,Image.ANTIALIAS)
    x_test.append(np.array(im))
y_test=[l.index(d[1]) for d in data_test]

x_test = np.stack(x_test)
y_test = np.array(y_test)


""" avec pickle, on sauvegarde les données dans un fichier pour ne pas avoir a retraiter le dataset"""

print("pickling")
f=open( "data_84x54_filtre_bruit_80_20.dtsim", "wb" )
pickle.dump([x_train,y_train,x_test,y_test,l],f,protocol=4)
f.close()

