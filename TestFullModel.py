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
from operator import itemgetter, attrgetter


from readDataset import LoadDataset, Interval2Segments, Segments2Data
from AutoEncoder import FullChannelEncoder, FullChannelDecoder
from LSTMmodel import LSTMLayer
from sklearn.model_selection import KFold
from scipy.signal import resample
import PreProcessing

import pyedflib


class ValidatonTestData :
    def __init__(self, interval_sets, window_size, model, info_file_path, edf_file_path, which_data):
        self.window_size = window_size
        self.model = model
        self.info_file_path = info_file_path
        self.edf_file_path = edf_file_path
        self.interval_sets = self.IntervalSorting(interval_sets)
        self.which_data = which_data
        self.full_signal = []
        self.target_sr = 128
        self.duration = 0
        self.alarm_interval = 10


    def IntervalSorting(self, interval_sets):
        state_list = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']
        state_dict = {'preictal_ontime':1, 'ictal':0, 'preictal_late':0, 'preictal_early':0, 'postictal':0,'interictal':0}
        temp = []
        for state in state_list:
            for interval in interval_sets[state]:
                temp += [(interval + [state_dict[state]])]
        
        temp = sorted(temp, key=itemgetter(0,1))
        return temp
    
    def LoadFileData(self, patient_name):
        # 환자 이름에 해당하는 EDF 파일을 self.full_signal 변수에 데이터 넣음
        channels_for_type = {
        'SNU': ['Fp1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'Fp2-AVG', 'F4-AVG', 'C4-AVG', 
                        'P4-AVG', 'F7-AVG', 'T1-AVG', 'T3-AVG', 'T5-AVG', 'O1-AVG', 'F8-AVG', 
                        'T2-AVG', 'T4-AVG', 'T6-AVG', 'O2-AVG', 'Fz-AVG', 'Cz-AVG', 'Pz-AVG'],
        'CHB': ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3',
                        'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4',
                        'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8',
                        'P8-O2', 'FZ-CZ', 'CZ-PZ']
        }
        file_path = self.GetFilePath(patient_name)
        with pyedflib.EdfReader(file_path) as f:
            labels = f.getSignalLabels()
            freq = f.getSampleFrequencies()
            self.duration = f.getFileDuration()
            self.full_signal = np.array([])
            for channel in channels_for_type[self.which_data]:
                ch_idx = labels.index(channel)
                edf_signal = f.readSignal(ch_idx)
                np.append(self.full_signal, (resample(edf_signal, int(len(edf_signal) / freq[ch_idx] * self.target_sr ))), axis=0)

    def MakeSegments(self):
        # segment = [start, duration]


    
    def GetFilePath(self, patient_name):
        return self.edf_file_path+'/'+(patient_name.split('_'))[0]+'/'+patient_name+'.edf'

#%%
if __name__=='__main__':
    window_size = 5
    overlap_sliding_size = 1
    normal_sliding_size = 1
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
        test_segments_set[state] = Interval2Segments(test_interval_set[state], edf_file_path, window_size, overlap_sliding_size)
        

    for state in ['postictal', 'interictal']:
        test_segments_set[state] = Interval2Segments(test_interval_set[state], edf_file_path, window_size, normal_sliding_size)

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

    x_data = Segments2Data(X_seg)
    x_data = PreProcessing.FilteringSegments(x_data)
    x_batch = np.split(x_data, 10, axis=-1) # (10, batch, eeg_channel, data)
    x_batch = np.transpose(x_batch,(1,0,2,3))

    original_data = x_data
    y_predict = fullmodel.predict(x_batch)


    




    