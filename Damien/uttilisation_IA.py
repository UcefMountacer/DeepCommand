# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:49:23 2019

@author: damie
"""

import tkinter as tk
import tkinter.filedialog as tkf
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
from extract_features import librosa_features, calcul_features
import sounddevice as sd
from scipy.io.wavfile import write
from data_augmentation import manipulate1,manipulate2,manipulate3
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'







import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import numpy as np

## Réglage de connection TCP/IP ###
TCP_Robot = "192.168.1.22"          # Adresse IP du robot
TCP_Mon_Ordi = "192.168.1.65"      # Adresse IP de mon ordi (Il faut paramétrer une liaison manuellement
                                    # et écrire une adresse IP dans le même sous réseau
                                    # que le robot 192.168.1.XX)
TCP_PORT_GRIPPER = 40001            # Ne pas changer
TCP_PORT = 30002                    # Ne pas changer

# Connection au robot
robot = urx.Robot(TCP_Robot)
robot.set_tcp((0, 0, 0.3, 0, 0, -0.43)) # Position du Tool Center Point par rapport au bout du robot (x,y,z,Rx,Ry,Rz)
                                    # (en mm et radians)
robot.set_payload(1, (0, 0, 0.1))       # Poids de l'outil et position de son centre de gravité (en kg et mm)

# Connection à la pince
gripper = Robotiq_Two_Finger_Gripper(robot)

# Caractéristique de mouvement
acc = 0.3                           # Accélération maximale des joints
vel = 0.3                           # Vitesse maximale des joints
deg2rad = np.pi/180
angular_pos = [-250*deg2rad, -55*deg2rad, 50*deg2rad, -90*deg2rad, 250*deg2rad, -50*deg2rad]














class my_IA():
    def __init__(self,model,knn,label):
        """
        on cree un objet qui regroupe le model et le knn pour faciliter son uttilisation
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



""" chargement des differents models """

#file=tkf.askopenfilename(filetypes=[("IA",".h5")])
model=tf.keras.models.load_model('Pre_knn_v4.h5')
model2 = tf.keras.models.Sequential()
for layer in model.layers[:-5]:
    model2.add(layer)
#file=tkf.askopenfilename(filetypes=[("IA",".ia")])
f=open('knn_v4.ia','rb')
knn,l=pickle.load(f)
f.close()


custom=my_IA(model2,knn,l)
IA_opperator=tf.keras.models.load_model('operator_image.h5')

""" definition des points importants """
point_outil_1_1=[286/1000., -424/1000., 318/1000., 0.733, 3.036, 0.025]
point_outil_1_2=[300/1000., -429/1000., 55/1000., 0.272, -3.177, 0.012]

point_outil_2_1=[469/1000., -411/1000., 300/1000., 0.2410, -3.165, -0.078]
point_outil_2_2=[469/1000., -411/1000., 49/1000., 0.2410, -3.165, -0.078]

point_operateur=[123/1000., -665/1000., 206/1000., 0.789, -3.128, 0.253]
point_house=[-275/1000., -117/1000., 398/1000., 0.34, 3.16, 0.05]


secu_table=156.84



class operator():
    def __init__(self,name=None,gender=None,gaucher=None,taille=None):
        self.name=name
        self.gender=gender
        self.gaucher=gaucher
        self.taille=taille
    def get_name(self):
        if self.name==None:
            print("No name defined")
            return ""
        else :
            return self.name
    def get_gender(self):
        if self.gender==None:
            print("No gender defined")
            return ""
        else :
            return self.gender
    def get_size(self):
        if self.taille==None:
            print("default 1m78")
            return 178
        else :
            return self.taille
    def is_gaucher(self):
        return self.gaucher

youssef=operator("youssef","M",True,178)
damien=operator("damien","F",False,165)
francois=operator("francois","M",False,178)


class net_wind(tk.Frame):
    """
    
    """
    def __init__(self):
        tk.Frame.__init__(self)
        
        
        
        """ chargement des differents models """

        #file=tkf.askopenfilename(filetypes=[("IA",".h5")])
        model=tf.keras.models.load_model('Pre_knn_v4.h5')
        model2 = tf.keras.models.Sequential()
        for layer in model.layers[:-5]:
            model2.add(layer)
        #file=tkf.askopenfilename(filetypes=[("IA",".ia")])
        f=open('knn_v4.ia','rb')
        knn,l=pickle.load(f)
        f.close()
        
        self.dic={}
        self.dic["on"]=self.on
        self.dic["one"]=self.one
        self.dic["two"]=self.two
        self.dic["go"]=self.go
        self.dic["no"]=self.go
        self.dic["right"]=self.right
        self.dic["left"]=self.left
        self.dic["off"]=self.off
        self.dic["tree"]=self.tree
        self.dic["three"]=self.tree
        self.dic["house"]=self.house
        self.dic["up"]=self.up
        self.dic["down"]=self.down
        self.model=my_IA(model2,knn,l)
        
        self.operator=None
        self.dic_operator={}
        self.dic_operator[0]=damien
        self.dic_operator[1]=youssef
        self.dic_operator[2]=francois
        
        
        
        
        """ creation des elements graphiques"""
        
        self.L=tk.Label(self,text="parlez quand la fenetre passe au vert")
        self.L.pack()
        #creation du bouton record
        self.B=tk.Button(self,text="record (green)",command=self.record)
        self.B.pack()
        
        self.configure(bg='red') #on met le fond de la fenetre en rouge
        
        self.fs = 44100  # Sample rate
        self.seconds = 1  # Duration of recording
        

        #on defini le model (IA a uttiliser)

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
#        res=file_2_tensor_d_aug("output.wav")
        op=IA_opperator(res)
        op=np.array(op)[0]
        print(op.argmax())
        self.operator=self.dic_operator[op]
        
        
        """
        on choisit l'action a effectuer en fonction du mot detecte
        
        up -> robot monte de 10cm
        down -> robot descend de 10cm
        right -> robot se deplace de 10cm vers la droite 
        left -> robot se deplace de 10cm vers la gauche
        go -> robot avance de 10cm 
        tree -> robot recule de 10cm
        
        one -> robot va chercher l'outil 1
        two -> robot va chercher l'outil 2
        house -> robot retourne a sa position initiale
        
        on -> affiche un message d'accueil
        off -> affiche un message d'au revoir
        """
        commande=l[mot[0]]
        if commande in self.dic :
            self.dic[commande]()

        self.configure(bg='red')
        self.update()
        return mot
    def up(self):
        robot.up(vel=vel, acc=acc)
    def down(self):
        robot.down(vel=vel, acc=acc)
    def right(self):
        robot.translate([0.1, 0, 0], acc=acc,vel=vel)
    def left(self):    
        robot.translate([-0.1, 0, 0], acc=acc,vel=vel)
    def go(self):
        robot.translate([0, -0.1, 0], acc=acc,vel=vel)
    def tree(self):
        robot.translate([0, 0.1, 0], acc=acc,vel=vel)
    def one(self):
        gripper.close_gripper() 
        robot.movel(point_outil_1_1, acc=acc, vel=vel)
        gripper.gripper_action(160)  
        robot.movel(point_outil_1_2, acc=acc, vel=vel)          
        gripper.close_gripper() 
        robot.movel(point_outil_1_1, acc=acc, vel=vel)
        
        
        h=self.operator.get_size*10*0.65
        hdefault=1780*0.65
        poscustom=point_operateur-[0,0,(hdefault-h)/1000.,0,0,0]
        if self.operator.is_gaucher():
            delta=(self.operator.get_size()*10*0.174/2)/1000.
            poscustom=poscustom+[delta,0,0,0,0,0]
        
        
        
        robot.movel(point_operateur, acc=acc, vel=vel)
        time.sleep(2)
        gripper.gripper_action(0)
            
        time.sleep(2)
        gripper.close_gripper() 
    def two(self):
        gripper.close_gripper() 
        robot.movel(point_outil_2_1, acc=acc, vel=vel)
        gripper.gripper_action(160)  
        robot.movel(point_outil_2_2, acc=acc, vel=vel)
        gripper.close_gripper() 
        robot.movel(point_outil_2_1, acc=acc, vel=vel)
        robot.movel(point_operateur, acc=acc, vel=vel)
        time.sleep(2)
        gripper.gripper_action(0)
        
        time.sleep(2)
        gripper.close_gripper() 
    def house(self):
        robot.movel(point_house, acc=acc, vel=vel)
    def on(self):
        print("Bonjour cher operateur ! ")
    def off(self):
        print("Au revoir !")
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
#    os.remove(file)
    os.remove('temp/output.png')
    
    return tf.keras.utils.normalize(tf.cast(np.stack([np.array(im)]), tf.float32), axis=1)
def file_2_tensor_y(file):
    """
    converti un fichier wav en tensor en utilisant les features de librosa
    """
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
    ys, sr = librosa.load(file)
    dictio = librosa_features(ys, sr)
    DF = calcul_features(dictio)
    
    DF=DF[columns]
    X=DF.values
#    X=scalar.transform(X)
def file_2_tensor_d_aug(file):
    left = 54
    top = 35
    width = 334
    height = 216
    box = (left, top, left+width, top+height)
    size= 84,54
    y0, sr = librosa.load(file)
    y1=manipulate1(y0,2)
    y2=manipulate2(y0,44100,2)
    y3=manipulate3(y0,2)
    y4=manipulate3(y0,0.7)
    Y=[y0,y1,y2,y3,y4]
    X=[]
    for y in Y :
        S, phase = librosa.magphase(librosa.stft(y=y))
        plt.figure(1)
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
        plt.savefig("temp/output.png")
        plt.close()
        im=Image.open("temp/output.png")
        im=im.crop(box)
        im.thumbnail(size,Image.ANTIALIAS)
        os.remove('temp/output.png')
        X.append(tf.cast(np.array(im), tf.float32))
    
    return X
    
    
    return X

class app(tk.Frame):
    """creation de la fenetre principale"""
    def __init__(self):
        tk.Frame.__init__(self)
        self.wind=net_wind()
        self.wind.pack()
        #self.iconbitmap('D:/bural/cours/projet/github/logo-couleur-cmjn-jpg.ico')
        self.update()



app().mainloop()