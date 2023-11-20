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
import PreProcessing


#%%
window_size = 5
overlap_sliding_size = 5
sr = 128
data_type = 'chb'
normal_sliding_size = window_size
state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

# for WSL
if data_type == 'snu':
    test_info_file_path = "/host/d/SNU_DATA/SNU_patient_info_test.csv"
    edf_file_path = "/host/d/SNU_DATA"
elif data_type == 'chb':
    test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
    edf_file_path = "/host/d/CHB"
    
test_interval_set = LoadDataset(test_info_file_path)
test_segments_set = {}

# 상대적으로 데이터 갯수가 적은 것들은 window_size 2초에 sliding_size 1초로 overlap 시켜 데이터 증강
for state in ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early']:
    test_segments_set[state] = Interval2Segments(test_interval_set[state],edf_file_path, window_size, overlap_sliding_size)
    

for state in ['postictal', 'interictal']:
    test_segments_set[state] = Interval2Segments(test_interval_set[state],edf_file_path, window_size, normal_sliding_size)

test_type_1 = np.array(test_segments_set['preictal_ontime'] + test_segments_set['preictal_early'] + test_segments_set['preictal_late'])
test_type_2 = np.array(test_segments_set['ictal'])
test_type_3 = np.array(test_segments_set['postictal'] + test_segments_set['interictal'])


#%%

input_type_1 = test_type_1[np.random.choice(len(test_type_1), 1, replace=False)]
input_type_2 = test_type_2[np.random.choice(len(test_type_2), 1, replace=False)]
input_type_3 = test_type_3[np.random.choice(len(test_type_3), 1, replace=False)]

x_batch_type_1 = Segments2Data(input_type_1, data_type)
x_batch_type_2 = Segments2Data(input_type_2, data_type)
x_batch_type_3 = Segments2Data(input_type_3, data_type)

x_batch = None

type_1_done = 0
type_2_done = 0
type_3_done = 0
if x_batch_type_1.ndim == 3:
    if np.all(x_batch== None):
        x_batch = x_batch_type_1
    else:
        x_batch = np.concatenate((x_batch, x_batch_type_1))
    type_1_done = 1

if x_batch_type_2.ndim == 3:
    if np.all(x_batch== None):
        x_batch = x_batch_type_2
    else:
        x_batch = np.concatenate((x_batch, x_batch_type_2))
    type_2_done = 1

if x_batch_type_3.ndim == 3:
    if np.all(x_batch== None):
        x_batch = x_batch_type_3
    else:
        x_batch = np.concatenate((x_batch, x_batch_type_3))
    type_3_done = 1

x_batch = PreProcessing.FilteringSegments(x_batch)


original_data = x_batch

original_data = np.squeeze(original_data)

print(input_type_2)

plt.figure(figsize=(48,27))
if type_1_done == 1 :
    plt.subplot(3,1,1)
    plt.plot(original_data[0][0],'b')
    plt.legend(labels=["Origin"])

if type_2_done == 1 :
    plt.subplot(3,1,2)
    plt.plot( original_data[1][0],'b')
    plt.legend(labels=["Origin"])

if type_3_done == 1 :
    plt.subplot(3,1,3)
    plt.plot(original_data[2][0],'b')

plt.legend(labels=["Origin"])

#freq = np.fft.fftfreq(len(original_data[0][0]),1/sr)


plt.show()
#plt.savefig(f"./ae_test/testfig.png")
# %%
