# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:32:40 2019

@author: Mountassir Youssef
"""
from matplotlib import pyplot as plt
import pandas as pd
#-------------------------------
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
#--------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


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
y=data['label'].values
X=preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X.shape, y.shape)

''' one hot encoding '''

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)
dummy_y.shape

''' train validation split '''

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.3, shuffle=True)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

''' define baseline model '''

# create model
model = Sequential()
model.add(Dense(100, activation='relu',kernel_initializer='random_normal', input_dim=60))
#model.add(Dense(100, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(200, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(30, activation='relu',kernel_initializer='random_normal'))
#model.add(Dense(30, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(10, activation='relu',kernel_initializer='random_normal'))
model.add(Dense(9, activation='softmax',kernel_initializer='random_normal'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

''' if ever wanted to use k-fold to have a more precise validation '''

estimator = KerasClassifier(build_fn=model, epochs=80, batch_size=100, verbose=0)
kfold = KFold(n_splits=4, shuffle=True, random_state=3)
cross_val_score(estimator, X, dummy_y, cv=4)

''' fitting train data '''

history=model.fit(X_train,y_train , epochs=100)

''' print the evolution of epochs '''
history

''' predicting validation data '''

y_pred=model.predict(X_test)

''' taking values higher than 0.5 '''

y_pred =(y_pred>0.5)

''' plotting the confusion matrix '''

from sklearn.metrics import confusion_matrix
cm = matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)
import seaborn as sn
x_axis_labels=['on','off','right','left','up','down','one','two','house']
sn.heatmap(cm, annot=True, xticklabels=x_axis_labels, yticklabels=x_axis_labels)

''' saving the model into an h5 file '''
model.save("model_command_Y_D_F.h5")

''' plottinh evolution of the training '''

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()