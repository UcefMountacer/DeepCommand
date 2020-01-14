# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:36:45 2019

@author: Mountassir Youssef
"""

import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


''' importing data to a dataframe'''
''' NB: the csv below is made for both command and operator neural networks '''

data = pd.read_csv('C:/Users/Mountassir Youssef/Desktop/PJE/Data/operator_train_with_augment_only_3.csv')

''' choosing the right columns in the dataframe '''

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

''' put data in arrays X and Y for standardization '''

X=data[columns].values
y=data['label_operator'].values
scalar = preprocessing.StandardScaler()
X=scalar.fit_transform(X.astype(float))

''' one hot encoding '''

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)
dummy_y.shape

''' train validation split '''

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.4, shuffle=True)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# define baseline model
X_train=scalar.fit_transform(X_train.astype(float))

''' define baseline model '''

# create model
model = Sequential()
model.add(Dense(100, activation='relu',kernel_initializer='random_normal', input_dim=60))
#model.add(Dense(100, activation='relu',kernel_initializer='random_normal'))
#model.add(Dense(200, activation='relu',kernel_initializer='random_normal'))
#"model.add(Dense(200, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(150, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(100, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(50, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(30, activation='softmax',kernel_initializer='random_normal'))
model.add(Dense(3, activation='softmax',kernel_initializer='random_normal'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

''' fitting train data '''

history=model.fit(X_train,y_train, epochs=100)
history 

''' saving the model '''

model.save("model_operator_detection_functional.h5")


''' predicting val data taking values higher than 0.5 '''

y_pred=(model.predict(X_test)>0.5)

''' plotting the confusion matrix '''

cm = tf.math.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1),num_classes=3)
print(cm)
import seaborn as sn
plt.figure()
x_axis_labels=['damien','youssef','françois']
sn.heatmap(cm, annot=True, xticklabels=x_axis_labels, yticklabels=x_axis_labels)
plt.show()

#y_pred =(y_pred>0.5)
#
##------- train set -----
#y_pred_train = model.predict(X_train)
#y_pred_train =(y_pred_train>0.5)
#train_accuracy = 100 - np.mean(np.abs(y_train - y_pred_train))*100
#test_accuracy = 100-np.mean(np.abs(y_test - y_pred))*100
#print('report')
#print('train_accuracy',train_accuracy)
#print('test_accuracy',test_accuracy)
#print('confusion matrix')
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
#print(cm)

#cm = tf.math.confusion_matrix().numpy()




#model.save('C:/Users/Mountassir Youssef/Desktop/PJE/final_model.h5')
#np.save('C:/Users/Mountassir Youssef/Desktop/PJE/moyenne',scalar.mean_)
#np.save('C:/Users/Mountassir Youssef/Desktop/PJE/var',scalar.var_)

#con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

#con_mat_df = pd.DataFrame(con_mat_norm,
#                     index = l, 
#                     columns = l)
#figure = plt.figure(figsize=(8, 8))
#sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
#plt.tight_layout()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()
#plt.savefig("matrice_de_confusion_v2.png")

#fichier='C:/Users/Mountassir Youssef/Desktop/PJE/Data/our_data_commands/damien/down/down_11.wav'
#
#
#
#
#
#def file_2_tensor_y(file):
#    ys, sr = librosa.load(file)
#    plt.plot(ys)
#    dictio = librosa_features(ys, sr)
#    DF = calcul_features(dictio)
##    DF=DF.drop(['label'],axis=1)
#    
#    DF=DF[columns]
#    X=DF.values
#    X=scalar.transform(X)
#    return X
#
#x=file_2_tensor_y(fichier)
#model.predict(x)






