# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:38:51 2019

@author: Youssef Mountassir

"""

'''
creating csv for training
'''

import csv

Train_Data = [['max_stft','min_stft','std_stft','amp_stft','mean_stft',
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
              'max_z_crossing','min_z_crossing','std_z_crossing','amp_z_crossing','mean_z_crossing','label','operator_label']]

with open('C:/Users/Mountassir Youssef/Desktop/PJE/Data/operator_train_with_augment_only_3.csv', 'w') as csvFile:
    writer = csv.writer(csvFile,delimiter=',')
    writer.writerows(Train_Data)
csvFile.close()

