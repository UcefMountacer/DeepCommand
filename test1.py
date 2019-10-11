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
import pickle

#files retrieval
EXT = "*.wav"
directory= "D:/bural/cours/projet/speech_commands_v0.01"
directory2= "D:/bural/cours/projet/test"


l=[os.path.basename(x[0]) for x in os.walk(directory)]
print(l[1:])
l.pop(0)
data=[]
for path, subdir, files in os.walk(directory):
    for file in glob(os.path.join(path, EXT)) :
        data.append([file,os.path.basename(path)])
print(data[0])
data_test=[]
for path, subdir, files in os.walk(directory2):
    for file in glob(os.path.join(path, EXT)) :
        data_test.append([file,os.path.basename(path)])


"""
direc = [file
                 for path, subdir, files in os.walk(directory)
                 for file in glob(os.path.join(path, EXT))]   
"""
#print(direc)
F = []
#load the files

#print(data[0][0])
#y, sr = librosa.load(data[0][0])
#S, phase = librosa.magphase(librosa.stft(y=y))
#librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
#plt.show()
#
#  







import tensorflow as tf

"""

#pre working on the data

x_train=[]
i=0
for d in data:
    
    y, sr = librosa.load(d[0])
    f1 = librosa.feature.chroma_stft(y,sr)
    x_train.append(f1)
    i+=1
    if i%1000==0 :
        print(i)

#x_train=[librosa.feature.chroma_stft(librosa.load(data[i][0])) for i in range(len(data))]
y_train=[l.index(d[1]) for d in data]

x_test=[]
i=0
for d in data_test:
    
    y, sr = librosa.load(d[0])
    f1 = librosa.feature.chroma_stft(y,sr)
    x_test.append(f1)
    i+=1
    if i%1000==0 :
        print(i)
y_test=[l.index(d[1]) for d in data_test]
#(x_train, y_train),(x_test, y_test) = mnist.load_data()
#print(x_train[0])


#reshape somme audio 

maxdiml=[]
for i in range(len(x_train)):
    maxdiml.append(np.size(x_train[i],1))

#    print(type(x_train[i]))
#    print(np.size(x_train[i],0))
#    print(np.size(x_train[i],1))

index_max= maxdiml.index(max(maxdiml))
max_size=np.size(x_train[index_max],1)
for i in range(len(x_train)):
    if np.size(x_train[i],1) != max_size:
        x_train[i] = np.append(x_train[i],np.zeros([12,max_size-np.size(x_train[i],1)]),axis=1)

maxdiml=[]
for i in range(len(x_test)):
    maxdiml.append(np.size(x_test[i],1))

#    print(type(x_train[i]))
#    print(np.size(x_train[i],0))
#    print(np.size(x_train[i],1))

index_max= maxdiml.index(max(maxdiml))
max_size=np.size(x_test[index_max],1)

for i in range(len(x_test)):
    if np.size(x_test[i],1) != max_size:
        x_test[i] = np.append(x_test[i],np.zeros([12,max_size-np.size(x_test[i],1)]),axis=1)

#convert lists into arrays

y_train = np.array(y_train)
x_train = np.stack(x_train)
#x_train = np.vstack(x_train)

x_test = np.stack(x_test)
y_test = np.array(y_test)

print(np.size(x_train,0))
print(np.size(x_train,1))
print(np.size(x_train,2))

#saving data

print("pickling")
f=open( "dataset.dts", "wb" )
pickle.dump([x_train,y_train,x_test,y_test,l],f)
f.close()
"""

#f=open("dataset.dts","rb")
#x_train,y_train,x_test,y_test,labels = pickle.load(f)
#f.close()


#print(labels)

f=open("D:/bural/cours/projet/data_image_small_2.dtsim", "rb" )
x_train,y_train,x_test,y_test,l=pickle.load(f)
f.close()


x_train = tf.keras.utils.normalize(x_train, axis=1)


import matplotlib.pyplot as plt

#x_train = tf.keras.utils.normalize(x_train,order=2, axis=1)
#x_test = tf.keras.utils.normalize(x_test,order=2, axis=1)

#print(x_train[0])

"""
https://towardsdatascience.com/boost-your-cnn-image-classifier-performance-with-progressive-resizing-in-keras-a7d96da06e20 
"""
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(54, 84, 4),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3),activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(tf.keras.layers.Conv2D(32, (3, 3),activation=tf.nn.relu))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3),activation=tf.nn.relu))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(30, activation=tf.nn.softmax))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=10)


#◙model.save("premier_testv4_bigerimg_RMSprop.h5")

#f=open( "data_test_image.dtsim", "rb" )
#x_test,y_test=pickle.load(f)
#f.close()
x_test = tf.keras.utils.normalize(x_test, axis=1)

val_loss, val_acc = model.evaluate(x_test, y_test,verbose=0)
print(val_loss)
print(val_acc)



