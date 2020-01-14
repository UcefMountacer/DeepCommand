# -*- coding: utf-8 -*-

import tkinter as tk
import matplotlib as plt
import numpy as np
import librosa
import librosa.display
import pickle
from extract_features import librosa_features, calcul_features
import time
import pandas as pd
import sounddevice as sd
import soundfile as sf
from keras.models import load_model
from data_augmentation import manipulate1,manipulate2,manipulate3
from sklearn import preprocessing


''' import model '''

model = load_model('C:/Users/Mountassir Youssef/Desktop/PJE/Data/RRRRR.h5')

''' import mean and var of the training data '''

mean = pickle.load(open('C:/Users/Mountassir Youssef/Desktop/PJE/Data/mean.pkl','rb'))
var = pickle.load(open('C:/Users/Mountassir Youssef/Desktop/PJE/Data/var.pkl','rb'))

''' use them to define the scaler '''

#scaler = preprocessing.StandardScaler()
#scaler.mean_=mean
#scaler.var_=var


columns = ['min_stft','std_stft','amp_stft','mean_stft',
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
              'max_z_crossing','min_z_crossing','std_z_crossing','amp_z_crossing']

class net_wind(tk.Frame):
    """
    
    """
    def __init__(self):
        tk.Frame.__init__(self)
        
        #creation du bouton record
        self.B=tk.Button(self,text="record (green)",command=self.record_and_predict)
        self.B.pack()
        
        #creation de l'entree avec mot ecrit dedant par defaut
        self.E=tk.Entry(self)
        self.E.insert(tk.END,"mot")
        self.E.pack()
        
        #creation du bouton pour changer de dossier
#        self.B2=tk.Button(self,text="change directory",command=self.chdir)
#        self.B2.pack()

        self.directory="C:/Users/Mountassir Youssef/Desktop/PJE/Data/tt" #dossier d'enregistrement des fichiers
        self.fs = 44100  # Sample rate
        self.seconds = 1  # Duration of recording
        self.configure(bg='red') #on met le fond de la fenetre en rouge
        
    def record_and_predict(self):
        """
        On enregistre le son pendant une seconde quand la fenetre passe au vert
        
        """
        time.sleep(0.1)
        self.configure(bg='green')#la fenetre passe au vert pour indiquer que l'enregistrement commence
        self.update()
        
        time.sleep(0.5) #on attend 0.1 seconde, c'est le temps de réaction d'un homme attentif
        myrecording = sd.rec(int(self.seconds * self.fs), samplerate=self.fs, channels=1)
        sd.wait()  # Wait until recording is finished
        
        self.configure(bg='cyan')#la fenetre passe au bleu pour indiquer que l'audio est en cour de traitement
        self.update()
        
        filename = 'C:/Users/Mountassir Youssef/Desktop/PJE/Data/audio_by_app.wav'
        
        sf.write(filename, myrecording, self.fs)
        
        file_2_pkl(filename)
        
        ARR = pkl_2_array('C:/Users/Mountassir Youssef/Desktop/PJE/Data/temp.pkl')
        ARR1 = pkl_2_array('C:/Users/Mountassir Youssef/Desktop/PJE/Data/temp1.pkl')
        ARR2 = pkl_2_array('C:/Users/Mountassir Youssef/Desktop/PJE/Data/temp2.pkl')
        ARR3 = pkl_2_array('C:/Users/Mountassir Youssef/Desktop/PJE/Data/temp3.pkl')
        ARR4 = pkl_2_array('C:/Users/Mountassir Youssef/Desktop/PJE/Data/temp4.pkl')
        
        pred = prediction_model(ARR)
        pred1 = prediction_model(ARR1)
        pred2 = prediction_model(ARR2)
        pred3 = prediction_model(ARR3)
        pred4 = prediction_model(ARR4)
        
        D=[pred[0][0],pred1[0][0],pred2[0][0],pred3[0][0],pred4[0][0]]
        Y=[pred[0][1],pred1[0][1],pred2[0][1],pred3[0][1],pred4[0][1]]
        F=[pred[0][2],pred1[0][2],pred2[0][2],pred3[0][2],pred4[0][2]]
        
        prediction=[np.mean(D),
                    np.mean(Y),
                    np.mean(F)]
        
        print(prediction)
        
        
        self.configure(bg='red')
        self.update()
        
#    def chdir(self):
#        """
#        Ouvre une fenetre pour choisir un nouveau dossier de travail
#        """
#        file=tkf.askopenfilename()
#        res=file_2_tensor_y(file)
#        pred = model.predict(res)
#        
#        print(pred)


class app(tk.Frame):
    def __init__(self):
        tk.Frame.__init__(self)
        self.wind=net_wind()
        self.wind.pack()
        self.update()
        
''' predict from an array '''

def prediction_model(Array):
    Prediction_Array=[]
    Prediction_Array=model.predict(Array)
    return Prediction_Array

''' convert picle to array '''

def pkl_2_array(path):
    d=pd.read_pickle(path)
    d=d[columns]
    X=d.values
#    Xd=scaler.transform(X)
#    Xd=(X-mean)/var
    return X

''' save the audio data in a pickle '''

def file_2_pkl(file):

    ys,sr=0,0
    ys, sr = librosa.load(file)
    #data augmnt
    y1=manipulate1(ys,2)
    y2=manipulate2(ys,44100,2)
    y3=manipulate3(ys,2)
    y4=manipulate3(ys,0.7)
    #
    dictio,dictio1,dictio2,dictio3,dictio4=0,0,0,0,0
    
    dictio = librosa_features(ys, sr)
    dictio1 = librosa_features(y1, sr)
    dictio2 = librosa_features(y2, sr)
    dictio3 = librosa_features(y3, sr)
    dictio4 = librosa_features(y4, sr/0.7)
    
    DF,DF1,DF2,DF3,DF4=0,0,0,0,0
    DF = calcul_features(dictio) 
    DF1 = calcul_features(dictio1)
    DF2 = calcul_features(dictio2)
    DF3 = calcul_features(dictio3)
    DF4 = calcul_features(dictio4)
      
    DF.to_pickle('C:/Users/Mountassir Youssef/Desktop/PJE/Data/temp.pkl')
    DF1.to_pickle('C:/Users/Mountassir Youssef/Desktop/PJE/Data/temp1.pkl')
    DF2.to_pickle('C:/Users/Mountassir Youssef/Desktop/PJE/Data/temp2.pkl')
    DF3.to_pickle('C:/Users/Mountassir Youssef/Desktop/PJE/Data/temp3.pkl')
    DF4.to_pickle('C:/Users/Mountassir Youssef/Desktop/PJE/Data/temp4.pkl')
    
    
''' testing '''    

#file = 'C:/Users/Mountassir Youssef/Desktop/PJE/Data/our_data_commands/youssef/down/down_1.wav'
#
#ys, sr = librosa.load(file)
#plt.plot(ys)
#dictio = librosa_features(ys, sr)
#DF = calcul_features(dictio) 
#
#d=DF[columns]
#X=d.values
#Xs=scalar.transform(X)
#
#hmm=model1.predict(Xs)
#hmm
#####################"


#bruit = "C:/Users/Mountassir Youssef/Desktop/PJE/free-spoken-digit-dataset-master/recordings/youssef/down/1.wav"
#r = file_2_tensor_y(bruit)
#type(r)


app().mainloop()





