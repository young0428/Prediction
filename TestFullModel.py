from datetime import datetime
import sys
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import pickle
import os

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence


from readDataset import LoadDataset, Interval2Segments, Segments2Data
from AutoEncoder import FullChannelEncoder, FullChannelDecoder
from LSTMmodel import LSTMLayer
from sklearn.model_selection import KFold
from PreProcessing import GetBatchIndexes


if __name__=='__main__':
    window_size = 20
    overlap_sliding_size = 1
    normal_sliding_size = 5
    state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

    # for WSL
    test_info_file_path = "/host/d/SNU_DATA/SNU_patient_info_test.csv"
    edf_file_path = "/host/d/SNU_DATA"

    # # for window
    # test_info_file_path = "D:/SNU_DATA/SNU_patient_info_test.csv"
    # edf_file_path = "D:/SNU_DATA"

    test_interval_set = LoadDataset(test_info_file_path)
    test_segments_set = {}

    # 상대적으로 데이터 갯수가 적은 것들은 window_size 2초에 sliding_size 1초로 overlap 시켜 데이터 증강
    for state in ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early']:
        test_segments_set[state] = Interval2Segments(test_interval_set[state],edf_file_path, window_size, overlap_sliding_size)
        

    for state in ['postictal', 'interictal']:
        test_segments_set[state] = Interval2Segments(test_interval_set[state],edf_file_path, window_size, normal_sliding_size)

    # type 1은 True Label데이터 preictal_ontime
    # type 2는 특별히 갯수 맞춰줘야 하는 데이터
    # type 3는 나머지


    test_type_1 = np.array(test_segments_set['preictal_ontime'])
    test_type_2 = np.array(test_segments_set['ictal'] + test_segments_set['preictal_early'] + test_segments_set['preictal_late'])
    test_type_3 = np.array(test_segments_set['postictal'] + test_segments_set['interictal'])

    checkpoint_path = "FullModel_training_0/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    fullmodel = tf.keras.models.load_model(checkpoint_path)

    input_type_1 = test_type_1[np.random.choice(len(test_type_1), 100, replace=False)]
    input_type_2 = test_type_2[np.random.choice(len(test_type_2), 100, replace=False)]
    input_type_3 = test_type_3[np.random.choice(len(test_type_3), 100, replace=False)]

    X_seg = np.concatenate((input_type_1,input_type_2,input_type_3))
    y_batch = np.concatenate( ( np.ones(len(input_type_1)), (np.zeros(len(input_type_2))), (np.zeros(len(input_type_3))) )  )
    y_batch = y_batch.tolist()
    y_batch = list(map(int,y_batch))
    y_batch = np.eye(2)[y_batch]

    X_data = Segments2Data(X_seg)
    x_batch = np.split(X_data, 10, axis=-1) # (10, batch, eeg_channel, data)
    x_batch = np.transpose(x_batch,(1,0,2,3))

    original_data = X_data
    #predict_y = fullmodel.predict(x_batch)
    #predict_y = tf.round(predict_y)

    y_predict = fullmodel.predict(x_batch)
    print(y_predict)
    print(y_batch)



    # original_data = np.squeeze(original_data)
    # reconstructed_output = np.squeeze(reconstructed_output)

    # plt.figure(figsize=(20,20))
    
    # plt.subplot(2,2,1)
    # rand_idx = random.randrange(0,100)
    # plt.plot(original_data[rand_idx][3],'b')
    # plt.plot(reconstructed_output[rand_idx][3],'r')
    # plt.legend(labels=["Input", "Recontructed"])

    # plt.subplot(2,2,2)
    # rand_idx = random.randrange(0,100)
    # plt.plot(original_data[rand_idx][3],'b')
    # plt.plot(reconstructed_output[rand_idx][3],'r')
    # plt.legend(labels=["Input", "Recontructed"])

    # plt.subplot(2,2,3)
    # rand_idx = random.randrange(100,200)
    # plt.plot(original_data[rand_idx][3],'b')
    # plt.plot(reconstructed_output[rand_idx][3],'r')
    # plt.legend(labels=["Input", "Recontructed"])

    # plt.subplot(2,2,4)
    # rand_idx = random.randrange(200,300)
    # plt.plot(original_data[rand_idx][3],'b')
    # plt.plot(reconstructed_output[rand_idx][3],'r')
    # plt.legend(labels=["Input", "Recontructed"])


    # plt.savefig("testfig.png")



    