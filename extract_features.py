# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:32:40 2019

@author: Mountassir Youssef
"""

import librosa
import numpy as np

def extract_features(audio, freq):
    
    '''
    in: fichier audio et la freq d'acquisition
    out: dictionnaire contenant les features (16)
    '''
    
    #magnitude and phase
    S, phase = librosa.magphase(librosa.stft(audio))
    stft = np.abs(librosa.stft(audio))
    # Spectral features
    f1 = librosa.feature.chroma_stft(audio, freq)
    f2 = librosa.feature.chroma_cqt(audio, freq)
    f3 = librosa.feature.chroma_cens(audio, freq)
    f4 = librosa.feature.mfcc(audio, freq)
    #---------------------------------------
    f5 = librosa.feature.rms(S=S)
    f6 = librosa.feature.spectral_centroid(audio, freq)
    #---------------------------------------
    f7 = librosa.feature.spectral_bandwidth(audio, freq)
    f8 = librosa.feature.spectral_contrast(S=S, sr=freq)
#    f9 = librosa.feature.spectral_flatness(S**2, power=1.0)
    f10 = librosa.feature.spectral_rolloff(audio, freq, roll_percent=0.1)
    f11 = librosa.feature.poly_features(S=stft, order=2)
    f12 = librosa.feature.tonnetz(librosa.effects.harmonic(audio), freq)
    f13 = librosa.feature.zero_crossing_rate(audio)
    

    
    dictionnaire = {}
    dictionnaire['f1']=f1
    dictionnaire['f2']=f2
    dictionnaire['f3']=f3
    dictionnaire['f4']=f4
    dictionnaire['f5']=f5
    dictionnaire['f6']=f6
    dictionnaire['f7']=f7
    dictionnaire['f8']=f8
    dictionnaire['f9']=0
    dictionnaire['f10']=f10
    dictionnaire['f11']=f11
    dictionnaire['f12']=f12
    dictionnaire['f13']=f13


    
    return dictionnaire