# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:31:35 2019

@author: Mountassir Youssef
"""
EXT = "*.wav"
import librosa
import numpy as np

def manipulate1(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def manipulate2(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def manipulate3(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)         

