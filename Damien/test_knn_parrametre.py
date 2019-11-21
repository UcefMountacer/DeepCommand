# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:43:08 2019

@author: damie
"""



import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


"""on importe le dataset"""
f=open("D:/bural/cours/projet/data_84x54_filtre_bruit_80_20.dtsim", "rb" )
x_train,y_train,x_test,y_test,l=pickle.load(f)
f.close()

"""on normalise les donnees"""
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


"""on importe notre model entraine"""
model=tf.keras.models.load_model("D:/bural/cours/projet/Pre_KNN.h5")


"""
on test quelle couche de notre model donne les meilleur features
pour le knn, on teste aussi differantes valeures de K pour 
trouver la valeure qui correspond le mieux a notre problemes
"""

print("model -1")
plt.figure(1)

"""on creer un nouveau model qui ne possedera qu'une partie du model entraine"""
model2 = tf.keras.models.Sequential()
"""
model -i correspond a model.layers[:-i] ce qui permet de tester quelle couche
donne les meilleures features
"""
for layer in model.layers[:-1]:
    model2.add(layer)


"""on cree les features"""
features=model2.predict(x_train)
X_test=model2.predict(x_test)


k_range=range(3,20)#on fait varier k entre 0 et 20
scores={}
score_list=[]
for k in k_range :
    knn=KNeighborsClassifier(n_neighbors=k)  # create the neural network
    knn.fit(features,y_train)  # train the neural network
    #on calcul ensuite le score du knn
    y_pred=knn.predict(X_test)
    scores[k]=metrics.accuracy_score(y_test,y_pred)
    score_list.append(metrics.accuracy_score(y_test,y_pred))
"""on affiche la courbe score en foction de k"""
plt.plot(k_range,score_list)

"""on repete ces opperations pour differents layers"""
print("model -2")
plt.figure(2)
model2 = tf.keras.models.Sequential()

for layer in model.layers[:-2]:
    model2.add(layer)



features=model2.predict(x_train)
X_test=model2.predict(x_test)

k_range=range(3,20)
scores={}
score_list=[]
for k in k_range :
    knn=KNeighborsClassifier(n_neighbors=k)  
    knn.fit(features,y_train)  
    y_pred=knn.predict(X_test)
    scores[k]=metrics.accuracy_score(y_test,y_pred)
    score_list.append(metrics.accuracy_score(y_test,y_pred))

plt.plot(k_range,score_list)

print("model -3")
plt.figure(3)
model2 = tf.keras.models.Sequential()

for layer in model.layers[:-3]:
    model2.add(layer)



features=model2.predict(x_train)
X_test=model2.predict(x_test)

k_range=range(3,20)
scores={}
score_list=[]
for k in k_range :
    knn=KNeighborsClassifier(n_neighbors=k)  
    knn.fit(features,y_train)  
    y_pred=knn.predict(X_test)
    scores[k]=metrics.accuracy_score(y_test,y_pred)
    score_list.append(metrics.accuracy_score(y_test,y_pred))

plt.plot(k_range,score_list)



print("model -4")
plt.figure(4)
model2 = tf.keras.models.Sequential()

for layer in model.layers[:-4]:
    model2.add(layer)



features=model2.predict(x_train)
X_test=model2.predict(x_test)

k_range=range(3,20)
scores={}
score_list=[]
for k in k_range :
    knn=KNeighborsClassifier(n_neighbors=k)  
    knn.fit(features,y_train)  
    y_pred=knn.predict(X_test)
    scores[k]=metrics.accuracy_score(y_test,y_pred)
    score_list.append(metrics.accuracy_score(y_test,y_pred))

plt.plot(k_range,score_list)

print("model -6")
plt.figure(6)
model2 = tf.keras.models.Sequential()

for layer in model.layers[:-6]:
    model2.add(layer)



features=model2.predict(x_train)
X_test=model2.predict(x_test)

k_range=range(3,20)
scores={}
score_list=[]
for k in k_range :
    knn=KNeighborsClassifier(n_neighbors=k)  
    knn.fit(features,y_train)  
    y_pred=knn.predict(X_test)
    scores[k]=metrics.accuracy_score(y_test,y_pred)
    score_list.append(metrics.accuracy_score(y_test,y_pred))

plt.plot(k_range,score_list)
plt.show()
