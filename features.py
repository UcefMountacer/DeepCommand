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
directory="C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/stop"

'''
d1= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/left"
d2= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/on"
d3= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d4= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/cat"
d5= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/dog"
d6= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/down"
d7= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/up"
d8= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/stop"
d9= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/no"
d10= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/yes"
d11= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/wow"
d12= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/on"
d13= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d14= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d15= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d16= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d17= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d18= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d19= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d20= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d21= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/left"
d22= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/on"
d23= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d24= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d25= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d26= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d27= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d28= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"
d29= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/Speech commands/off"

D = [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29]


for d in D:
'''
direc = [file
                 for path, subdir, files in os.walk(directory)
                 for file in glob(os.path.join(path, EXT))] 
  
#load the files and extract features

for i in range(len(direc)):
    y, sr = librosa.load(direc[i])
    S, ph = librosa.magphase(librosa.stft(y))
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr)
    dictio = librosa_features(y, sr)
    DF = calcul_features(dictio)
    DF['label']=9
    with open('C:/Users/Mountassir Youssef/Desktop/PJE/Data/data.csv','a',newline='') as csvFile:
            DF.to_csv(csvFile, header=False)
    csvFile.close() 


    
    
    
    
    
    