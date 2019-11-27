# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:36:45 2019

@author: Mountassir Youssef
"""

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import math
from keras import layers
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('C:/Users/Mountassir Youssef/Desktop/PJE/Data/operator_train.csv')


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
X=scalar.fit(X).transform(X.astype(float))
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
model.add(Dense(80, activation='relu',kernel_initializer='random_normal', input_dim=60))
#model.add(Dense(100, activation='relu',kernel_initializer='random_normal'))
#model.add(Dense(200, activation='relu',kernel_initializer='random_normal'))
#"model.add(Dense(200, activation='relu',kernel_initializer='random_normal'))
#model.add(Dense(100, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(30, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(20, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(6, activation='softmax',kernel_initializer='random_normal'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train,y_train, epochs=60)
history 
y_pred=model.predict(X_test)
y_pred =(y_pred>0.5)

#------- train set -----
y_pred_train = model.predict(X_train)
y_pred_train =(y_pred_train>0.5)
train_accuracy = 100 - np.mean(np.abs(y_train - y_pred_train))*100
train_accuracy
test_accuracy = 100-np.mean(np.abs(y_test - y_pred))*100
test_accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(cm)
import seaborn as sn
sn.heatmap(cm, annot=True)

model.save("C:/Users/Mountassir Youssef/Desktop/PJE/models/model_operator_detection.h5")









