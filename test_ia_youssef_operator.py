# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:49:23 2019

@author: damie
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:19:37 2019

@author: damie
"""
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
from PIL import Image
import random as rd
from PIL import ImageTk as itk
from playsound import playsound
from sklearn import preprocessing
import pandas as pd
#from numba import cuda
#print(cuda.gpus)
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import sounddevice as sd
from scipy.io.wavfile import write



data=pd.read_csv('data_youssef/operator_train.csv')
data=data.drop(['label_operator'],axis=1)
Xd=data.values
scalar = preprocessing.StandardScaler()
Xd=scalar.fit(Xd)







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

        self.directory="D:/bural/cours/projet/dataset_moi" #dossier d'enregistrement des fichiers
        self.fs = 44100  # Sample rate
        self.seconds = 1  # Duration of recording
        self.configure(bg='red') #on met le fond de la fenetre en rouge

        file=tkf.askopenfilename(filetypes=[("IA",".ia"),("IA",".h5")])
        
        self.model=tf.keras.models.load_model(file)
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
        
        
        print(self.model.predict(res))
        
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


def file_2_tensor_y(file):
    t1=time.time()
    y, sr = librosa.load(file)
#    S, ph = librosa.magphase(librosa.stft(y))
#    S_mel = librosa.feature.melspectrogram(y=y, sr=sr)
    dictio = librosa_features(y, sr)
    DF = calcul_features(dictio)
    DF=DF.drop(['label'],axis=1)
    
    X=DF.values
    X=scalar.transform(X)
    #X=preprocessing.StandardScaler().fit(X).transform(X)
#    f=open("arrayM.np",'rb')
#    M=pickle.load(f)
#    f.close()
#    f=open("arrayV.np",'rb')
#    V=pickle.load(f)
#    f.close()
#    print(X)
#    X=np.divide((X-M),V)
#    print(X)
#    X=tf.keras.utils.normalize(X)
    print(time.time()-t1,"=============")
    return X
app().mainloop()