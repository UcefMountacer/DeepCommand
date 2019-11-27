# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:33:12 2019

@author: Mountassir Youssef
"""


import librosa
import os
from glob import glob
import librosa.display
from extract_features import librosa_features, calcul_features



EXT = "*.wav"
#directory="C:/Users/Mountassir Youssef/Desktop/PJE/free-spoken-digit-dataset-master/recordings/youssef/down/1.wav"
bruit = "C:/Users/Mountassir Youssef/Desktop/PJE/free-spoken-digit-dataset-master/recordings/youssef/down/1.wav"



#direc = [file
#                 for path, subdir, files in os.walk(directory)
#                 for file in glob(os.path.join(path, EXT))] 
#import noise
yb, srb = librosa.load(bruit)
#load the files and extract features
#
#for i in range(len(direc)):
#    y, sr = librosa.load(direc[i])
#    taille = y.shape[0]
#    yb_cut = yb[0:taille]
#    #superposition
#    y=y+yb_cut
#    S, ph = librosa.magphase(librosa.stft(y))
#    S_mel = librosa.feature.melspectrogram(y=y, sr=sr)
#    dictio = librosa_features(y, sr)
#    DF = calcul_features(dictio)
#    DF['label']=1
#    with open('C:/Users/Mountassir Youssef/Desktop/PJE/Data/classification_super_bruit.csv','a',newline='') as csvFile:
#            DF.to_csv(csvFile, header=False)
#    csvFile.close() 


    
    
    
    
    
    