from datetime import datetime
import sys
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns
from readDataset import *

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from operator import itemgetter, attrgetter


from readDataset import LoadDataset, Interval2Segments, Segments2Data, MakePath, IntervalList2Dict
from AutoEncoder import FullChannelEncoder, FullChannelDecoder
from LSTMmodel import LSTMLayer
from sklearn.model_selection import KFold
from scipy.signal import resample
import PreProcessing
from sklearn.metrics import confusion_matrix

import pyedflib


class ValidatonTestData :
    def __init__(self, interval_sets, window_size, model, batch_size, info_file_path, edf_file_path, which_data, channel, preprocessing):
        self.window_size = window_size
        self.model = model
        self.info_file_path = info_file_path
        self.edf_file_path = edf_file_path
        self.interval_sets = interval_sets
        self.which_data = which_data
        self.full_signal = []
        self.target_sr = 200
        self.duration = 0
        self.alarm_interval = 10
        self.cat_num = 2
        self.matrix = np.zeros(shape=(self.cat_num, self.cat_num))
        self.tf_matrix = np.zeros(shape=(2,2))
        self.batch_size = batch_size
        self.patient_dict = {}
        self.seg_res = []
        self.channel = channel
        self.preprocessing = preprocessing

    def start(self,k,n):
        sorted_intervals = self.IntervalSorting(self.interval_sets) # [환자명, start, end, state_label]
        self.sorted_intervals = sorted_intervals
        self.SetKN(k,n)
        self.MakeSegments(self.sorted_intervals)
        batch_idx_seq = self.GetBatchIndexes()
        for idx, batch_idx in enumerate(batch_idx_seq):
            self.Segments2Data(self.segments[batch_idx], self.labels[batch_idx])
            self.Predict()
            self.PostProcessing()
            self.Result2Mat()
            sens, far = self.Calc()
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
        state_list = ['preictal_ontime','interictal']
        self.state_dict = {'preictal_ontime':1, 'interictal':0}
        temp = []
        self.intervals_state = []
        
        for state in state_list:
            if state in interval_sets.keys():
                for interval in interval_sets[state]:
                    temp += [(interval[:3] + [self.state_dict[state]] + [state])]
                    self.intervals_state.append(state)
        
        temp = sorted(temp, key=itemgetter(0,1))
        # [환자명, start, end, state_label]
        return temp
    
    def LoadFileData(self, patient_name):
        # 환자 이름에 해당하는 EDF 파일을 self.full_signal 변수에 데이터 넣음
        channels_for_type = {
        'snu': ['Fp1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'Fp2-AVG', 'F4-AVG', 'C4-AVG', 
                        'P4-AVG', 'F7-AVG', 'T1-AVG', 'T3-AVG', 'T5-AVG', 'O1-AVG', 'F8-AVG', 
                        'T2-AVG', 'T4-AVG', 'T6-AVG', 'O2-AVG', 'Fz-AVG', 'Cz-AVG', 'Pz-AVG'],
        'chb': ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3',
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
                # resampling하여 target_sr로 다운 샘플링
                self.full_signal.append((resample(edf_signal, int(len(edf_signal) / freq[ch_idx] * self.target_sr )))) # (21, target_sr * file_duration)
            self.full_signal = np.array(self.full_signal)
    def MakeSegments(self, sorted_interval):
        # segment = [start, duration]
        self.sliding_size = 2
        start_gap = max(0, self.window_size - self.sliding_size)
        time_for_k_of_n = self.sliding_size * self.k + start_gap
        # inverval = [환자명, start, end, state_label]
        self.segments = []
        self.labels = []
        self.segments_state = []
        for interval in sorted_interval :
            time = float(interval[1]) # start

            while not (time + time_for_k_of_n > float(interval[2])) : # end
                self.segments.append([ MakePath(interval[0],self.edf_file_path) ,time, time_for_k_of_n, interval[4]])  # segment = [start, duration]
                self.labels.append(interval[3])
                self.segments_state.append(interval[4])
                time += self.alarm_interval
        self.segments = np.array(self.segments)
        self.labels = np.array(self.labels)
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
        self.results = []
        
        # segment = [filepath, start, duration]
        for seg_idx, segment in enumerate(segments):
            
            signal = Segments2Data([segment],self.which_data, self.channel)
            signal = signal[0]
            #signal /= 50
            start_idx = 0
            sample_num = int(self.target_sr * self.window_size)
            sliding_num = int(self.target_sr * self.sliding_size)
            for j in range(self.k):
                s = start_idx + j * sliding_num
                if self.preprocessing == None:
                    x = signal[:,s:s+sample_num]
                else:
                    
                    x = self.preprocessing(signal[:,s:s+sample_num],200,64)
                    x = np.squeeze(x)
                    
                    
                self.batch_x.append(x)
                self.true.append(labels[seg_idx])
                self.results.append([segment[0], float(segment[1]) + j*self.sliding_size,self.window_size, segment[3]])
                
    # 생성된 배칭에 대해 예측 수행
    def Predict(self):
        
        self.batch_x = np.expand_dims(self.batch_x,axis=-1)
        self.predict = self.model.predict_on_batch(self.batch_x)
        #self.pred_cat = ((tf.squeeze(tf.round(self.predict))).numpy()).astype(int).tolist()
        self.pred_cat = [1 if output[1] >= 0.5 else 0 for output in self.predict]

        for idx, segment in enumerate(self.results):
            self.results[idx] += [self.true[idx], self.predict[idx][1]]

            self.seg_res.append(self.results[idx])

    # K of N을 적용해서 alarm 울림 (1), alarm 안울림 (0) 결정
    # 데이터 
    def PostProcessing(self):
        self.true_k_of_n = []
        self.pred_k_of_n = []
        
        for i in range(int(len(self.pred_cat)/self.k)):
            true_cnt = 0
            # category 0 == preictal, category 1 == ictal, category3 = post, interictal
            # category0 이면 1(True)로 category1 or 2 이면 0(False)으로
            if self.true[i*self.k] == 1 :
                self.true_k_of_n.append(1)
            else:
                self.true_k_of_n.append(0)
            # Prediction 결과에 K of N 수행
            for j in range(self.k):
                if self.pred_cat[i*self.k+j] == 1 :
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
        if  not (self.matrix[1,0] + self.matrix[1,1]) == 0:
            sensitivity = self.matrix[1,1] / (self.matrix[1,0] + self.matrix[1,1])
        else:
            sensitivity = 0
        false_alarm_rate =  self.matrix[0,1] / ((self.matrix[0,0] + self.matrix[0,1]))
        return sensitivity, false_alarm_rate
    
    def SetKN(self,k,n):
        self.k = k
        self.n = n
        
    
    def GetFilePath(self, patient_name):
        return self.edf_file_path+'/'+(patient_name.split('_'))[0]+'/'+patient_name+'.edf'

#%%

def validation(checkpoint_path,test_interval_set, data_type,k,n, window_size, channel, preprocessing = None):
    overlap_sliding_size = 1
    normal_sliding_size = 1
    test_batch_size = 10


    # for WSL
    #lstm_model_name = "paper_base_rawEEG_categorical"
    if data_type == 'snu' or data_type == 'snu_one_ch':
        # test_info_file_path = "/host/d/SNU_DATA/patient_info_snu_test.csv"
        # edf_file_path = "/host/d/SNU_DATA"
        test_info_file_path = "/host/d/SNU_DATA/patient_info_snu_test.csv"
        edf_file_path = "/host/d/SNU_DATA"
    elif data_type == 'chb' or data_type == 'chb_one_ch':
        # test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
        # edf_file_path = "/host/d/CHB"

        test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
        edf_file_path = "/host/d/CHB"
    



    checkpoint_dir = os.path.dirname(checkpoint_path)
    fullmodel = tf.keras.models.load_model(checkpoint_path)

    val_object = ValidatonTestData(test_interval_set, window_size, fullmodel, test_batch_size, test_info_file_path, edf_file_path, data_type, channel, preprocessing)
    # %%
    val_object.start(k,n)
    sens,fpr = val_object.Calc()
    del fullmodel
    return val_object.matrix, val_object.tf_matrix, sens, fpr, val_object.seg_res

# lstm_model_name = "one_ch_dilation_lstm_300sec_random_FP1-F7"
# window_size = 300
# patient_name = "CHB014"
# idx = 1
# checkpoint_path = f"./Dilation/{lstm_model_name}/{patient_name}/set{idx+1}/cp.ckpt"
# interval_sets = [['CHB014_22', 0, 399, 'interictal'], ['CHB014_03', '2000', '3059', 'preictal_early'], ['CHB014_03', '3059', '3600', 'preictal_ontime'], ['CHB014_04', '0', '1252', 'preictal_ontime'], ['CHB014_04', '1252', '1372', 'preictal_late'], ['CHB014_04', '1372', '1392', 'ictal'], ['CHB014_14', '1858', '3259', 'interictal']]

# interval_sets = IntervalList2Dict(interval_sets)
# validation(checkpoint_path, interval_sets, 'chb_one_ch', 1,1, window_size=window_size, channel=['FP1-F7'])

def SaveAsHeatmap(matrix, path):
    sns.heatmap(matrix,annot=True, cmap='Blues')
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.savefig(path)
    plt.clf()



    
# %%
