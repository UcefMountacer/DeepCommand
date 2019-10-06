# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:47:49 2019

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

#files retrieval
EXT = "*.wav"
directory= "D:/bural/cours/projet/speech_commands_v0.01"
directory2= "D:/bural/cours/projet/test"
directory3= "D:/bural/cours/projet/train_img"
directory4= "D:/bural/cours/projet/test_img"


l=[os.path.basename(x[0]) for x in os.walk(directory)]
print(l[1:])

#code pour creer les dossiers ou seront stockees les images
#for lab in l :
#    os.makedirs(directory4+"/"+lab)

""" on met tous les fichiers et le nom de leur dosssier (cat,dog,...) dans une liste data et data_test """
data=[]
for path, subdir, files in os.walk(directory):
    for file in glob(os.path.join(path, EXT)) :
        data.append([file,os.path.basename(path)])
print(data[0])
data_test=[]
for path, subdir, files in os.walk(directory2):
    for file in glob(os.path.join(path, EXT)) :
        data_test.append([file,os.path.basename(path)])


"""
direc = [file
                 for path, subdir, files in os.walk(directory)
                 for file in glob(os.path.join(path, EXT))]   
"""
#print(direc)
F = []
""" on converti les fichier audios en images puis on les sauvegardes """

for i in range(27931,len(data)):
    y, sr = librosa.load(data[i][0])
    S, phase = librosa.magphase(librosa.stft(y=y))
    plt.figure(1)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
    plt.savefig(directory3+"/"+data[i][1]+"/"+os.path.basename(data[i][0])[:-4]+".png")
    plt.close()
    if i%1000 == 0 :
        gc.collect()
for i in range(26845,len(data_test)):
    y, sr = librosa.load(data_test[i][0])
    S, phase = librosa.magphase(librosa.stft(y=y))
    plt.figure(1)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
    plt.savefig(directory4+"/"+data_test[i][1]+"/"+os.path.basename(data_test[i][0])[:-4]+".png")
    plt.close()
    if i%1000 == 0 :
        gc.collect()