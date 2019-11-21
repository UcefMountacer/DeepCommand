# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:49:23 2019

@author: damie
"""

import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
import librosa.display
import pickle
import time
from PIL import Image
from sklearn import preprocessing
import sounddevice as sd
from scipy.io.wavfile import write
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


class my_IA():
    def __init__(self,model,knn,label):
        """
        on creer un objet qui regroupe le model et le knn pour faciliter son uttilisation
        """
        self.model=model
        self.knn=knn
        self.label=label
    def predict(self , new_input):
        self.out1=self.model.predict(new_input)
        self.out_index=self.knn.predict(self.out1)
        self.out_label=[self.label[self.out_index[i]] for i in range(len(self.out_index))]
        print(self.out_label)
        return(self.out_index)






model=tf.keras.models.load_model("Pre_KNN.h5")
model2 = tf.keras.models.Sequential()
for layer in model.layers[:-5]:
    model2.add(layer)
f=open("knn_trained.ia",'rb')
knn,l=pickle.load(f)
f.close()
custom=my_IA(model2,knn,l)








class net_wind(tk.Frame):
    """
    
    """
    def __init__(self):
        tk.Frame.__init__(self)
        
        
        self.L=tk.Label(self,text="parlez quand la fenetre passe au vert")
        self.L.pack()
        #creation du bouton record
        self.B=tk.Button(self,text="record (green)",command=self.record)
        self.B.pack()
        
        self.fs = 44100  # Sample rate
        self.seconds = 1  # Duration of recording
        self.configure(bg='red') #on met le fond de la fenetre en rouge

        #on defini le model (IA a uttiliser)
        self.model=custom
    def record(self):
        """
        On enregistre le son pendant une seconde quand la fenetre passe au vert
        
        """
        time.sleep(0.65)
        self.configure(bg='green')#la fenetre passe au vert pour indiquer que l'enregistrement commence
        self.update()
        
        time.sleep(0.1) #on attend 0.1 seconde, c'est le temps de réaction d'un homme attentif
        myrecording = sd.rec(int(self.seconds * self.fs), samplerate=self.fs, channels=2)#on enregistre pendant 1s
        sd.wait()  # Wait until recording is finished
        
        self.configure(bg='cyan')#la fenetre passe au bleu pour indiquer que l'audio est en cour de traitement
        self.update()
        
        
        
        write("output.wav", self.fs, myrecording)  # Save as WAV file 
        
        res=file_2_tensor_d("output.wav")#on le converti en tenseur pour le model

        
        mot=self.model.predict(res)
        print(mot)
        self.configure(bg='red')
        self.update()
        return mot
        

def file_2_tensor_d(file):
    """
    converti un fichier wav en tensor (memes operations que dans les fichiers
    convertion wav-png et traitement d'image)
    """
    left = 54
    top = 35
    width = 334
    height = 216
    box = (left, top, left+width, top+height)
    size= 84,54
    y, sr = librosa.load(file)
    S, phase = librosa.magphase(librosa.stft(y=y))
    plt.figure(1)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
    plt.savefig("temp/output.png")
    plt.close()
    im=Image.open("temp/output.png")
    im=im.crop(box)
    im.thumbnail(size,Image.ANTIALIAS)
    os.remove(file)
    os.remove('temp/output.png')
    
    return tf.keras.utils.normalize(tf.cast(np.stack([np.array(im)]), tf.float32), axis=1)


class app(tk.Frame):
    """creation de la fenetre principale"""
    def __init__(self):
        tk.Frame.__init__(self)
        self.wind=net_wind()
        self.wind.pack()
        self.update()



app().mainloop()