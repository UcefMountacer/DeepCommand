# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:33:12 2019

@author: Mountassir Youssef
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from glob import glob
import librosa.display

#files retrieval
EXT = "*.wav"
directory= "C:/Users/Mountassir Youssef/Desktop/PJE/Data"

direc = [file
                 for path, subdir, files in os.walk(directory)
                 for file in glob(os.path.join(path, EXT))]   
F = []
#load the files
for i in range(len(direc)):
    y, sr = librosa.load(direc[i])
    f1 = librosa.feature.chroma_stft(y,sr)
    plt.figure(i)
    librosa.display.specshow(f1,y_axis='chroma', x_axis='time')
    plt.show()