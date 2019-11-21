# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:36:32 2019

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

"""creation des adresses des dossiers"""
EXT = "*.wav"
directory= "D:/bural/cours/projet/speech_commands_v0.01"
directory2= "D:/bural/cours/projet/test"

directory3= "D:/bural/cours/projet/train_img"
directory4= "D:/bural/cours/projet/test_img"


l=[os.path.basename(x[0]) for x in os.walk(directory)]
print(l[1:])


#code pour creer les dossiers ou seront stockees les images s'il ne sont pas deja cree
#for lab in l :
#    os.makedirs(directory4+"/"+lab)

""" on met tous les fichiers et le nom de leur dosssier (cat,dog,...) dans une liste data et data_test """
data=[]
for path, subdir, files in os.walk(directory):
    for file in glob(os.path.join(path, EXT)) :

        data.append([file,os.path.basename(path)])#[adresse du fichier,nom du dossier]
print(data[0])
data_test=[]
for path, subdir, files in os.walk(directory2):
    for file in glob(os.path.join(path, EXT)) :
        data_test.append([file,os.path.basename(path)])


""" 
on converti les fichier audios en images puis on les sauvegardes 
il faut l'executer plusieurs fois car la memoire RAM se remplie

pour relancer le programe, je vais dans le dossier ou les fichiers png
sont sauvegardes et je compte le nombre de fichier (selectionne tous les
dossier puis clic-droit propriete et nombre de fichiers)

on remplace "for i in range(len(data)):"
par "for i in range(nombres de fichiers,len(data)):"
"""

for i in range(len(data)):
    #on charge le fichier wav avec librosa
    y, sr = librosa.load(data[i][0])
    S, phase = librosa.magphase(librosa.stft(y=y))
    plt.figure(1)
    #on affiche le spectrogram
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
    #on sauvegarde l'image et on ferme le plot pour ne pas saturer la console
    plt.savefig(directory3+"/"+data[i][1]+"/"+os.path.basename(data[i][0])[:-4]+".png")
    plt.close()
    if i%1000 == 0 :
        """
        tout les 1000 fichiers on appel le garbage collector
        cela permet de vider un peu la memoire et donc de traiter
        plus de fichiers par execution
        """
        gc.collect()
for i in range(len(data_test)):
    y, sr = librosa.load(data_test[i][0])
    S, phase = librosa.magphase(librosa.stft(y=y))
    plt.figure(1)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
    plt.savefig(directory4+"/"+data_test[i][1]+"/"+os.path.basename(data_test[i][0])[:-4]+".png")
    plt.close()
    if i%1000 == 0 :
        gc.collect()