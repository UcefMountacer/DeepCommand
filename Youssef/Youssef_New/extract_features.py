# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:32:40 2019

@author: Mountassir Youssef
"""

import librosa
import numpy as np
import pandas as pd


def librosa_features(audio, freq):
    
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

    '''
    
    R = [f1,f2,f3,f4,f5,f6,f7,f8,f10,f11,f12,f13]
    return R
    '''


def calcul_features(D):
    
    """
    Input: D est le dictionnaire contenant les matrices calculées par Librosa
    
    Output: data frame, ou csv qui contient les features finaux
    
    """
    
    #get matrices
    
    
    stft = D['f1']
    cqt = D['f2']
    cens = D['f3']
    mfcc = D['f4']
    rms = D['f5']
    centroid = D['f6']
    bandwidth = D['f7']
    contrast = D['f8']
    rolloff = D['f10']
    poly = D['f10']
    tonnetz = D['f11']
    z_crossing = D['f12']
    '''
    stft = D[0]
    cqt = D[2]
    cens = D[3]
    mfcc = D[4]
    rms = D[5]
    centroid = D[6]
    bandwidth = D[7]
    contrast = D[8]
    rolloff = D[9]
    poly = D[10]
    tonnetz = D[11]
    z_crossing = D[12]
    '''
    #calculate variables of these matrices
    
    features = pd.DataFrame(index=['mot'],    columns = ['max_stft','min_stft','std_stft','amp_stft','mean_stft',
                                                      'max_cqt','min_cqt','std_cqt','amp_cqt','mean_cqt',
                                                      'max_cens','min_cens','std_cens','amp_cens','mean_cens',
                                                      'max_mfcc','min_mfcc','std_mfcc','amp_mfcc','mean_mfcc',
                                                      'max_rms','min_rms','std_rms','amp_rms','mean_rms',
                                                      'max_centroid','min_centroid','std_centroid','amp_centroid','mean_centroid',
                                                      'max_bandwidth','min_bandwidth','std_bandwidth','amp_bandwidth','mean_bandwidth',
                                                      'max_contrast','min_contrast','std_contrast','amp_contrast','mean_contrast',
                                                      'max_rolloff','min_rolloff','std_rolloff','amp_rolloff','mean_rolloff',
                                                      'max_poly','min_poly','std_poly','amp_poly','mean_poly',
                                                      'max_tonnetz','min_tonnetz','std_tonnetz','amp_tonnetz','mean_tonnetz',
                                                      'max_z_crossing','min_z_crossing','std_z_crossing','amp_z_crossing','mean_z_crossing','label','label_operator'])
    #
    
    features.max_stft = np.max(stft)
    features.min_stft = np.min(stft)
    features.std_stft = np.std(stft)
    features.amp_stft = np.max(stft) - np.min(stft)
    features.mean_stft = np.mean(stft)
    
    features.max_cqt = np.max(cqt)
    features.min_cqt = np.min(cqt)
    features.std_cqt = np.std(cqt)
    features.amp_cqt = np.max(cqt) - np.min(cqt)
    features.mean_cqt = np.mean(cqt)
    
    features.max_cens = np.max(cens)
    features.min_cens = np.min(cens)
    features.std_cens = np.std(cens)
    features.amp_cens = np.max(cens) - np.min(cens)
    features.mean_cens = np.mean(cens)
    
    features.max_mfcc = np.max(mfcc)
    features.min_mfcc = np.min(mfcc)
    features.std_mfcc = np.std(mfcc)
    features.amp_mfcc = np.max(mfcc) - np.min(mfcc)
    features.mean_mfcc = np.mean(mfcc)
    
    features.max_rms = np.max(rms)
    features.min_rms = np.min(rms)
    features.std_rms = np.std(rms)
    features.amp_rms = np.max(rms) - np.min(rms)
    features.mean_rms = np.mean(rms)
    
    features.max_centroid = np.max(centroid)
    features.min_centroid = np.min(centroid)
    features.std_centroid = np.std(centroid)
    features.amp_centroid = np.max(centroid) - np.min(centroid)
    features.mean_centroid = np.mean(centroid)
    
    features.max_bandwidth = np.max(bandwidth)
    features.min_bandwidth = np.min(bandwidth)
    features.std_bandwidth = np.std(bandwidth)
    features.amp_bandwidth = np.max(bandwidth) - np.min(bandwidth)
    features.mean_bandwidth = np.mean(bandwidth)
    
    features.max_contrast = np.max(contrast)
    features.min_contrast = np.min(contrast)
    features.std_contrast = np.std(contrast)
    features.amp_contrast = np.max(contrast) - np.min(contrast)
    features.mean_contrast = np.mean(contrast)
    
    features.max_rolloff = np.max(rolloff)
    features.min_rolloff = np.min(rolloff)
    features.std_rolloff = np.std(rolloff)
    features.amp_rolloff = np.max(rolloff) - np.min(rolloff)
    features.mean_rolloff = np.mean(rolloff)
    
    features.max_poly = np.max(poly)
    features.min_poly = np.min(poly)
    features.std_poly = np.std(poly)
    features.amp_poly = np.max(poly) - np.min(poly)
    features.mean_poly = np.mean(poly)
    
    features.max_tonnetz = np.max(tonnetz)
    features.min_tonnetz = np.min(tonnetz)
    features.std_tonnetz = np.std(tonnetz)
    features.amp_tonnetz = np.max(tonnetz) - np.min(tonnetz)
    features.mean_tonnetz = np.mean(tonnetz)
    
    features.max_z_crossing = np.max(z_crossing)
    features.min_z_crossing = np.min(z_crossing)
    features.std_z_crossing = np.std(z_crossing)
    features.amp_z_crossing = np.max(z_crossing) - np.min(z_crossing)
    features.mean_z_crossing = np.mean(z_crossing)
    features.label = 0
    features.label_operator = 0
    
    
    return features
    
    
    
#def data_augment(d1,d2,output_d):
#    y1,sr1 = librosa.load(d1)
#    y2,sr2 = librosa.load(d2)
#    sr = max(sr1,sr2)
#    y = (y1 + y2)/2
#    librosa.output.write_wav(output_d,y,sr)
    
    
    
    
    
    
    
    
    
    
    
    
    
    