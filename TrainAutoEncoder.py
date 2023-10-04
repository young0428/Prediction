from readDataset import LoadDataset, Interval2Segments, Segments2Data
from AutoEncoder import FullChannelEncoder, FullChannelDecoder
from LSTMmodel import LSTMLayer
import numpy as np

window_size = 2
overlap_sliding_size = 1
normal_sliding_size = window_size
state = ['ictal', 'preictal_late', 'preictal_early', 'preictal_ontime', 'postictal','interictal']

train_info_file_path = "D:/SNU_DATA/patient_info_train.csv"
test_info_file_path = "D:/SNU_DATA/patient_info_test.csv"
edf_file_path = "D:/SNU_DATA"


train_interval_set = LoadDataset(train_info_file_path)
train_segments_set = {}
# 상대적으로 데이터 갯수가 적은 것들은 window_size 2초에 sliding_size 1초로 overlap 시켜 데이터 증강
for state in ['ictal','preictal_ontime', 'preictal_late', 'preictal_early']:
    train_segments_set[state] = Interval2Segments(train_interval_set[state],edf_file_path, window_size, overlap_sliding_size)

for state in ['postictal', 'interictal']:
    train_segments_set[state] = Interval2Segments(train_interval_set[state],edf_file_path, window_size, normal_sliding_size)




    





