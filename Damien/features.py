
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
import gc
import time


EXT = "*.wav"
directory= "D:/bural/cours/projet/test"

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
l=[os.path.basename(x[0]) for x in os.walk(directory)]
l.pop(0)

l=["zero","one","two","three","four","five","six","seven","eight","nine","tree","go","house"]
l2=[13,14,15,16,17,18,19,20,21,22,23,24,25]



data=[]
for path, subdir, files in os.walk(directory):
    for file in glob(os.path.join(path, EXT)) :
        if os.path.basename(path) in l:
            data.append([file,l2[l.index(os.path.basename(path))]])

ld=len(data)
t1=time.time()
for i in range(ld):
    y, sr = librosa.load(data[i][0])
    S, ph = librosa.magphase(librosa.stft(y))
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr)
    dictio = librosa_features(y, sr)
    DF = calcul_features(dictio)
    DF['label']=data[i][1]
    with open('D:/bural/cours/projet/data_youssef/data.csv','a',newline='') as csvFile:
            DF.to_csv(csvFile, header=False)
    csvFile.close() 
    if i%500 == 0 :
        del(y)
        del(sr)
        del(S)
        del(ph)
        del(S_mel)
        del(dictio)
        del(DF)
        gc.collect()
        print(i,"/",ld)
        print(time.time()-t1)
        t1=time.time()