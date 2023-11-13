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
    def __init__(self, interval_sets, window_size, model, batch_size, info_file_path, edf_file_path, which_data):
        self.window_size = window_size
        self.model = model
        self.info_file_path = info_file_path
        self.edf_file_path = edf_file_path
        self.interval_sets = interval_sets
        self.which_data = which_data
        self.full_signal = []
        self.target_sr = 128
        self.duration = 0
        self.alarm_interval = 10
        self.cat_num = 3
        self.matrix = np.zeros(shape=(self.cat_num, self.cat_num))
        self.tf_matrix = np.zeros(shape=(2,2))
        self.batch_size = batch_size
        self.patient_dict = {}

    def start(self):
        sorted_intervals = self.IntervalSorting(self.interval_sets) # [환자명, start, end, state_label]
        self.sorted_intervals = sorted_intervals
        patient_name_list = self.GetPatientName(sorted_intervals)
        self.SetKN(5,3)
        for patient in patient_name_list:
            self.LoadFileData(patient)
            self.MakeSegments(self.patient_dict[patient])
            batch_idx_seq = self.GetBatchIndexes()
            for idx, batch_idx in enumerate(batch_idx_seq):
                val_object.Segments2Data(val_object.segments[batch_idx])
                val_object.Predict()
                val_object.PostProcessing()
                val_object.Result2Mat()
                sens, far = val_object.Calc()
                print(f'\rTest Progress {"%.2f"%((idx+1)/len(batch_idx_seq)*100)}% ({idx+1}/{len(batch_idx_seq)})   Sensitivity : {"%.2f"%(sens*100)}    FAR : {"%.4f"%(far)}', end='')
            
            print("")
            


    def GetPatientName(self, intervals):
        patient_name_list = []
        for interval in intervals:
            if not interval[0] in patient_name_list:
                patient_name_list.append(interval[0])
                self.patient_dict[interval[0]] = []
            self.patient_dict[interval[0]].append(interval)
        return patient_name_list

    def IntervalSorting(self, interval_sets):
        state_list = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']
        state_dict = {'preictal_ontime':0, 'ictal':1, 'preictal_late':0, 'preictal_early':0, 'postictal':2,'interictal':2}
        temp = []
        for state in state_list:
            for interval in interval_sets[state]:
                temp += [(interval + [state_dict[state]])]
        
        temp = sorted(temp, key=itemgetter(0,1))
        # [환자명, start, end, state_label]
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
            self.full_signal = []
            for idx, channel in enumerate(channels_for_type[self.which_data]):
                ch_idx = labels.index(channel)
                edf_signal = f.readSignal(ch_idx)
                self.full_signal.append((resample(edf_signal, int(len(edf_signal) / freq[ch_idx] * self.target_sr )))) # (21, target_sr * file_duration)
                print(f'\r{patient_name}, channel {channel} ({idx+1}/{len(channels_for_type[self.which_data])}) loaded...', end='')
            print("")
            self.full_signal = np.array(self.full_signal)
    def MakeSegments(self, sorted_interval):
        # segment = [start, duration]
        self.sliding_size = self.alarm_interval / self.k
        start_gap = max(0, self.window_size - self.sliding_size)

        # inverval = [환자명, start, end, state_label]
        self.segments = []
        self.labels = []
        
        for interval in sorted_interval :
            time = interval[1] # start
            while not (time + start_gap + self.alarm_interval > interval[2]) : # end
                self.segments.append([ time, start_gap + self.alarm_interval])  # segment = [start, duration]
                self.labels.append(interval[3])
                time += self.sliding_size 
        self.segments = np.array(self.segments)
    # 전체 segments를 batch_size로 나눔 
    def GetBatchIndexes(self):
        lst = list(range(len(self.segments)))
        idx_list = [lst[i:i+self.batch_size] for i in range(0, len(lst), self.batch_size)]
    
        return idx_list

    # segment에 있는 정보를 기반으로 메모리(self.full_signal)에 있는 실제 데이터를 가져옴
    # 같은 idx에 label 저장 ( categorical )
    def Segments2Data(self, segments, labels):
        self.batch_x = []
        self.true = []
        # segment = [start, duration]
        for seg_idx, segment in enumerate(segments):
            start_idx = int(self.target_sr * segment[0])
            sample_num = int(self.target_sr * self.window_size)
            sliding_num = int(self.target_sr * self.sliding_size)
            for j in range(self.k):
                s = start_idx + j * sliding_num
                self.batch_x.append(self.full_signal[:,s:s+sample_num])
                self.true.append(self.labels[seg_idx])

    def Predict(self):
        self.batch_x = np.expand_dims(self.batch_x,axis=-1)
        self.predict = self.model.predict(self.batch_x, verbose=0)
        self.pred_cat = np.argmax(self.predict,axis=1)

    # K of N을 적용해서 alarm 울림 (1), alarm 안울림 (0) 결정
    # 데이터 
    def PostProcessing(self):
        self.true_k_of_n = []
        self.pred_k_of_n = []
        
        for i in range(int(len(self.pred_cat)/self.k)):
            true_cnt = 0
            # category 0 == preictal, category 1 == ictal, category3 = post, interictal
            # category0 이면 1(True)로 category1 or 2 이면 0(False)으로
            if self.true[i*self.k] == 0 :
                self.true_k_of_n.append(1)
            else:
                self.true_k_of_n.append(0)
            # Prediction 결과에 K of N 수행
            for j in range(self.k):
                if self.pred_cat[i*self.k+j] == 0 :
                    true_cnt += 1

            if true_cnt >= self.n:
                self.pred_k_of_n.append(1) 
            else:
                self.pred_k_of_n.append(0)

    def Result2Mat(self):
        for idx in range(len(self.pred_cat)):
            self.matrix[self.true[idx],self.pred_cat[idx]] += 1 
        
        for idx in range(len(self.true_k_of_n)):
            self.tf_matrix[self.true_k_of_n[idx],self.pred_k_of_n[idx]] += 1 

    def Calc(self):
        sensitivity = self.tf_matrix[1,1] / (self.tf_matrix[1,0] + self.tf_matrix[1,1])
        false_alarm_rate =  self.tf_matrix[0,1] * ( 3600 / ((self.tf_matrix[0,0] + self.tf_matrix[0,1]) * self.alarm_interval))
        return sensitivity, false_alarm_rate
    
    def SetKN(self,k,n):
        self.k = k
        self.n = n
        
    
    def GetFilePath(self, patient_name):
        return self.edf_file_path+'/'+(patient_name.split('_'))[0]+'/'+patient_name+'.edf'

#%%
#def validation(lstm_model_name):
window_size = 5
overlap_sliding_size = 1
normal_sliding_size = 1
test_batch_size = 100

state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

# for WSL
lstm_model_name = "paper_base_rawEEG_categorical"
test_info_file_path = "/host/d/SNU_DATA/SNU_patient_info_test.csv"
edf_file_path = "/host/d/SNU_DATA"

# # for window
# test_info_file_path = "D:/SNU_DATA/SNU_patient_info_test.csv"
# edf_file_path = "D:/SNU_DATA"

checkpoint_path = f"LSTM/{lstm_model_name}/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

fullmodel = tf.keras.models.load_model(checkpoint_path)

test_interval_set = LoadDataset(test_info_file_path)
val_object = ValidatonTestData(test_interval_set, 5, fullmodel, test_batch_size, test_info_file_path, edf_file_path, 'SNU')
# %%
# val_object.start()
sorted_intervals = val_object.IntervalSorting(val_object.interval_sets) # [환자명, start, end, state_label]
print(sorted_intervals[0:5])
patient_name_list = val_object.GetPatientName(sorted_intervals)
print(patient_name_list[0:5])
val_object.SetKN(5,3)
#%%

val_object.LoadFileData(patient_name_list[0])
print(f'signal shape : {np.shape(val_object.full_signal)}')
#%%
val_object.MakeSegments(val_object.patient_dict[patient_name_list[0]])
print(f'segment shape : {np.shape(val_object.segments)}')
#%%   
batch_idx_seq = val_object.GetBatchIndexes()

for idx, batch_idx in enumerate(batch_idx_seq):
    val_object.Segments2Data(val_object.segments[batch_idx])

    val_object.Predict()
    val_object.PostProcessing()
    val_object.Result2Mat()
    sens, far = val_object.Calc()
    print(f'\rTest Progress {"%.2f"%((idx+1)/len(batch_idx_seq)*100)}% ({idx+1}/{len(batch_idx_seq)})   Sensitivity : {"%.2f"%(sens*100)}    FAR : {"%.4f"%(far)}', end='')
    #print(f'batch_x shape : {np.shape(val_object.batch_x)}')
    #print(f'true    shape : {np.shape(val_object.true)}')
    #print(f'Predict : {val_object.predict[0:25]}')

#%%

#val_object.start()

#validation("paper_base_rawEEG_categorical")

    # # 상대적으로 데이터 갯수가 적은 것들은 window_size 2초에 sliding_size 1초로 overlap 시켜 데이터 증강
    # for state in ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early']:
    #     test_segments_set[state] = Interval2Segments(test_interval_set[state], edf_file_path, window_size, overlap_sliding_size)
        

    # for state in ['postictal', 'interictal']:
    #     test_segments_set[state] = Interval2Segments(test_interval_set[state], edf_file_path, window_size, normal_sliding_size)

    # # type 1은 True Label데이터 preictal_ontime
    # # type 2는 특별히 갯수 맞춰줘야 하는 데이터
    # # type 3는 나머지


    # test_type_1 = np.array(test_segments_set['preictal_ontime'])
    # test_type_2 = np.array(test_segments_set['ictal'] + test_segments_set['preictal_early'] + test_segments_set['preictal_late'])
    # test_type_3 = np.array(test_segments_set['postictal'] + test_segments_set['interictal'])

    # checkpoint_path = "FullModel_training_0/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # fullmodel = tf.keras.models.load_model(checkpoint_path)

    # input_type_1 = test_type_1[np.random.choice(len(test_type_1), 100, replace=False)]
    # input_type_2 = test_type_2[np.random.choice(len(test_type_2), 100, replace=False)]
    # input_type_3 = test_type_3[np.random.choice(len(test_type_3), 100, replace=False)]

    # X_seg = np.concatenate((input_type_1,input_type_2,input_type_3))
    # y_batch = np.concatenate( ( np.ones(len(input_type_1)), (np.zeros(len(input_type_2))), (np.zeros(len(input_type_3))) )  )
    # y_batch = y_batch.tolist()

    # x_data = Segments2Data(X_seg)
    # x_data = PreProcessing.FilteringSegments(x_data)
    # x_batch = np.split(x_data, 10, axis=-1) # (10, batch, eeg_channel, data)
    # x_batch = np.transpose(x_batch,(1,0,2,3))

    # original_data = x_data
    # y_predict = fullmodel.predict(x_batch)


    




    
# %%
