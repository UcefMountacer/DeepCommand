# -*- coding: utf-8 -*-

import tkinter as tk
import tkinter.filedialog as tkf
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
from glob import glob
import librosa.display
import pickle
from extract_features import librosa_features, calcul_features
import gc
import time
#from PIL import Image
#import random as rd
#from PIL import ImageTk as itk
#from playsound import playsound
from sklearn import preprocessing
import pandas as pd
import sounddevice as sd
from scipy.io.wavfile import write
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# training 

data = pd.read_csv('C:/Users/Mountassir Youssef/Desktop/PJE/Data/ucef_train.csv')


columns = ['max_stft','min_stft','std_stft','amp_stft','mean_stft',
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
              'max_z_crossing','min_z_crossing','std_z_crossing','amp_z_crossing','mean_z_crossing']
X=data[columns].values
y=data['label_operator'].values
scalar = preprocessing.StandardScaler()
X=scalar.fit_transform(X.astype(float))

# one hot encoding
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)
dummy_y.shape
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.3, shuffle=True)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# define baseline model

# create model
model = Sequential()
model.add(Dense(30, activation='relu',kernel_initializer='random_normal', input_dim=60))
#model.add(Dense(100, activation='relu',kernel_initializer='random_normal'))
#model.add(Dense(200, activation='relu',kernel_initializer='random_normal'))
#"model.add(Dense(200, activation='relu',kernel_initializer='random_normal'))
#model.add(Dense(100, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(30, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(20, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(2, activation='softmax',kernel_initializer='random_normal'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,y_train, epochs=30)









#☻22222222222222222222222222222222222222222222222222222222222222



# real time test

class net_wind(tk.Frame):
    """
    
    """
    def __init__(self):
        tk.Frame.__init__(self)
        
        #creation du bouton record
        self.B=tk.Button(self,text="record (green)",command=self.record)
        self.B.pack()
        
        #creation de l'entree avec mot ecrit dedant par defaut
        self.E=tk.Entry(self)
        self.E.insert(tk.END,"mot")
        self.E.pack()
        
        #creation du bouton pour changer de dossier
        self.B2=tk.Button(self,text="change directory",command=self.chdir)
        self.B2.pack()

        self.directory="C:/Users/Mountassir Youssef/Desktop/PJE/Data/tt" #dossier d'enregistrement des fichiers
        self.fs = 44100  # Sample rate
        self.seconds = 2  # Duration of recording
        self.configure(bg='red') #on met le fond de la fenetre en rouge
        
    def record(self):
        """
        On enregistre le son pendant une seconde quand la fenetre passe au vert
        
        """
        time.sleep(0.65)
        self.configure(bg='green')#la fenetre passe au vert pour indiquer que l'enregistrement commence
        self.update()
        
        time.sleep(0.1) #on attend 0.1 seconde, c'est le temps de réaction d'un homme attentif
        myrecording = sd.rec(int(self.seconds * self.fs), samplerate=self.fs, channels=2)
        sd.wait()  # Wait until recording is finished
        
        self.configure(bg='cyan')#la fenetre passe au bleu pour indiquer que l'audio est en cour de traitement
        self.update()
        
        
        
        write("output.wav", self.fs, myrecording)  # Save as WAV file 
        
        res=file_2_tensor_y("output.wav")
        
        
        
        pred = model.predict(res)
        
        print(pred)
        
        self.configure(bg='red')
        self.update()
        
    def chdir(self):
        """
        Ouvre une fenetre pour choisir un nouveau dossier de travail
        """
        self.directory=tkf.askdirectory()


class app(tk.Frame):
    def __init__(self):
        tk.Frame.__init__(self)
        self.wind=net_wind()
        self.wind.pack()
        self.update()



# 22222222222222222222222222222222222222222222222222222222



def file_2_tensor_y(file):
    ys, sr = librosa.load(file)
    plt.plot(ys)
#    S, ph = librosa.magphase(librosa.stft(y))
#    S_mel = librosa.feature.melspectrogram(y=y, sr=sr)
    dictio = librosa_features(ys, sr)
    DF = calcul_features(dictio)
#    DF=DF.drop(['label'],axis=1)
    
    DF=DF[columns]
    X=DF.values
    X=scalar.transform(X)
    return X

#bruit = "C:/Users/Mountassir Youssef/Desktop/PJE/free-spoken-digit-dataset-master/recordings/youssef/down/1.wav"
#r = file_2_tensor_y(bruit)
#type(r)



app().mainloop()









