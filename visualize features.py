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
from extract_features import extract_features

#files retrieval
EXT = "*.wav"
directory= "C:/Users/Mountassir Youssef/Desktop/PJE/Data/test"

direc = [file
                 for path, subdir, files in os.walk(directory)
                 for file in glob(os.path.join(path, EXT))] 
  
#load the files and extract features
for i in range(len(direc)):
    y, sr = librosa.load(direc[i])
    S, ph = librosa.magphase(librosa.stft(y))
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr)
    dictio = extract_features(y, sr)
    
    # les chroma
    plt.subplot(7,2,1)
    librosa.display.specshow(dictio['f1'],y_axis='chroma', x_axis='time')
    plt.title('chroma stft')
    plt.subplot(7,2,2)
    librosa.display.specshow(dictio['f2'],y_axis='chroma', x_axis='time')
    plt.title('chroma cqt')
    plt.subplot(7,2,3)
    librosa.display.specshow(dictio['f3'],y_axis='chroma', x_axis='time')
    plt.title('chroma cens')
    
    #mel spectogramm
    plt.subplot(7,2,4)
    librosa.display.specshow(librosa.power_to_db(S_mel,ref=np.max),y_axis='mel',x_axis='time')
    plt.title('MEL spectogramm')
    
    #mfcc
    plt.subplot(7,2,5)
    librosa.display.specshow(dictio['f4'],x_axis='time')
    plt.title('MFCC')
    
    #rms
    plt.subplot(7,2,6)
    plt.semilogy(dictio['f5'].T, label='RMS Energy')
    plt.title('RMS energy')
    
    #log power spectogramm
    plt.subplot(7,2,7)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='log', x_axis='time')
    plt.title('log power spectogramm')
    
    #spectral centroid
    plt.subplot(7,2,8)
    plt.semilogy(dictio['f6'].T, label='Spectral centroid')
    plt.title('Spectral centroid')
    
    #spectral bandwidth
    plt.subplot(7,2,9)
    plt.semilogy(dictio['f7'].T, label='Spectral bandwidth')
    plt.title('Spectral bandwidth')
    
    #spectral contrast
    plt.subplot(7,2,10)
    librosa.display.specshow(dictio['f8'], x_axis='time')
    plt.title('Spectral contrast')
    
    #rolloff
    plt.subplot(7,2,11)
    plt.semilogy(dictio['f10'].T, label='Roll-off frequency')
    plt.title('roll off')
    
    #quadratic polynome
    plt.subplot(7,2,12)
    plt.plot(dictio['f11'][2], label='order=2', alpha=0.8)
    plt.title('quadratic poly')
    
    #tonal centroid
    plt.subplot(7,2,13)
    librosa.display.specshow(dictio['f12'], y_axis='tonnetz')
    plt.title('tonal centroid')
    
    
    
    
    
    
    