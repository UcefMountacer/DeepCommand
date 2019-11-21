# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:27:26 2019

@author: damie
"""

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
import seaborn as sns
import pickle
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt





""" importation du dataset pre-traite pour le model """
f=open("D:/bural/cours/projet/data_84x54_filtre_bruit_80_20.dtsim", "rb" )
x_train,y_train,x_test,y_test,label = pickle.load(f)
f.close()


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)



""" 
creation du model : les multiples layers a 128 neurones permettent de 
tester plusieurs features pour le K-nearest algorithm 
"""

model = tf.keras.models.Sequential()


model.add(tf.keras.layers.Conv2D(64, (4, 4), input_shape=(54, 84, 4),activation='relu'))

model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3),activation=tf.nn.relu))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3),activation=tf.nn.relu))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(65, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(26, activation=tf.nn.softmax))
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


"""entrainement du model"""
model.fit(x_train, y_train, epochs=15)

"""sauvegarde du model"""
model.save("Pre_KNN.h5")



"""calcule de la precision du model et de la matrice de confusion"""
val_loss, val_acc = model.evaluate(x_test, y_test,verbose=0)
print(val_loss)
print(val_acc)


y_pred=model.predict_classes(x_test)

con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm,
                     index = label, 
                     columns = label)
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
plt.savefig("matrice_de_confusion.png")