# Interface-Homme-Machine
Academic Project

# Rérérences

Audio Classification avec convolutional neural network
https://medium.com/@CVxTz/audio-classification-a-convolutional-neural-network-approach-b0a4fce8f6c

https://www.youtube.com/watch?v=_nOu_CHogWw

Sound Classification using Deep Learning (using librosa)
https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7

Data augmentation:
https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6

GitHub sur data augmentation pour capteurs inertiels:
https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data

plusieurs datasets : 
https://towardsdatascience.com/a-data-lakes-worth-of-audio-datasets-b45b88cd4ad
https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

Tensorflow speech command example : 
https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md

# Application IA

Pour charger un réseau de neurone, sauvegarder le model avec model.save("non du fichier .ia")

Pour charger un dataset (déja traité pour etre envoyé dans le réseau) :

  f=open("Nom du fichier.dts", "wb" )
  
  pickle.dump([x_train,y_train,x_test,y_test,l],f,protocol=4) (l étant la liste des labels [bed,bird,cat...])
  
  f.close()
  
Il peut lire des fichier audio .wav (j'en ai mis 2 dans le .rar crées avec audacity) de 1s (ou plus si l'IA a été entrainé sur des fichiers de plus de 1s)

Il peut enregistrer le son du micro et (essayer de) reconnaitre le mot

On peut aussi visualiser l'architecture du réseau



