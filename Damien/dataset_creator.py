# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:19:37 2019

@author: damie
"""
import tkinter as tk
import tkinter.filedialog as tkf
import os
from glob import glob
import sounddevice as sd
from scipy.io.wavfile import write
import time


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
        
        
        index=1
        flag=False
        
        
        for path, subdir, files in os.walk(self.directory):
            for file in glob(os.path.join(path, "*.wav")) :#on cherche tout les fichiers wav dans le dossier de travail
                y=os.path.basename(file).split("_")
                name=y[0] 
                num=int(y[1][:-4]) #on separe le mot et son ocurence, le [:-4] permet de retirer l'extension du fichier
                if str(self.E.get())==name and not flag:#test si le mot est deja present dans le dossier

                    while flag==False or index > 1000:#on cherche l'ocurence du mot
                        L=[int(os.path.basename(file2).split("_")[1][:-4]) if (os.path.basename(file2).split("_")[0] == name) else 0 for file2 in glob(os.path.join(path, "*.wav"))]
                        #print(L,index)
                        if index not in L :#si l'ocurence trouvee (indexe) n'est pas dans la liste des ocurences, on a trouve l'ocurence du mot
                            flag=True
                        else :
                            index+=1
#                    print(name,num)
        if not flag : #le mot n'existe pas dan le dossier 
            index=1
        outname=self.directory+"/"+self.E.get()+"_"+str(index)+".wav"
        print(outname)
        
        write(outname, self.fs, myrecording)  # Save as WAV file 
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



app().mainloop()