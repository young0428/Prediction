from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model

from readDataset import LoadDataset, Interval2Segments, Segments2Data
from AutoEncoder import FullChannelEncoder, FullChannelDecoder
from LSTMmodel import LSTMLayer

import numpy as np
import random
import operator


########### Prepare Dataset ###########
window_size = 2
overlap_sliding_size = 1
normal_sliding_size = window_size
state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

train_info_file_path = "D:/SNU_DATA/patient_info_train.csv"
test_info_file_path = "D:/SNU_DATA/patient_info_test.csv"
edf_file_path = "D:/SNU_DATA"


train_interval_set = LoadDataset(train_info_file_path)
train_segments_set = {}

test_interval_set = LoadDataset(test_info_file_path)
test_segments_set = {}

# 상대적으로 데이터 갯수가 적은 것들은 window_size 2초에 sliding_size 1초로 overlap 시켜 데이터 증강
for state in ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early']:
    train_segments_set[state] = Interval2Segments(train_interval_set[state],edf_file_path, window_size, overlap_sliding_size)
    test_segments_set[state] = Interval2Segments(test_interval_set[state],edf_file_path, window_size, overlap_sliding_size)
    

for state in ['postictal', 'interictal']:
    train_segments_set[state] = Interval2Segments(train_interval_set[state],edf_file_path, window_size, normal_sliding_size)
    test_segments_set[state] = Interval2Segments(test_interval_set[state],edf_file_path, window_size, normal_sliding_size)

# type 1은 True Label데이터 preictal_ontime
# type 2는 특별히 갯수 맞춰줘야 하는 데이터
# type 3는 나머지

# AutoEncoder 단계에서는 1:1:3

train_type_1 = train_segments_set['preictal_ontime']
train_type_2 = train_segments_set['ictal'] + train_segments_set['preictal_early'] + train_segments_set['preictal_late']
train_type_3 = train_segments_set['postictal'] + train_segments_set['interictal']

test_type_1 = test_segments_set['preictal_ontime']
test_type_2 = test_segments_set['ictal'] + test_segments_set['preictal_early'] + test_segments_set['preictal_late'] 
test_type_3 = test_segments_set['postictal'] + test_segments_set['interictal']

train_type_1_num = len(train_type_1)
train_type_3_sampled = random.sample(train_type_3, int(train_type_1_num/5))
train_type_3_sampled.sort(key=operator.itemgetter(0,1))
print(train_type_3_sampled[0:300])

train_selected = train_type_1 + train_type_2 + train_type_3_sampled
train_dataset = Segments2Data(train_selected)
print(train_dataset.shape)

########### Model Set ###########




# print(test_type_1.shape)
# print(test_type_2.shape)
# print(test_type_3.shape)















    





