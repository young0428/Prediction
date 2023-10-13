# %%
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model

from readDataset import LoadDataset, Interval2Segments, Segments2Data
from AutoEncoder import FullChannelEncoder, FullChannelDecoder, FullChannelEncoder_test, FullChannelDecoder_test
from LSTMmodel import LSTMLayer
from sklearn.model_selection import KFold
from PreProcessing import GetBatchIndexes

import sys
import numpy as np
import random
import operator
import matplotlib as plt

# %%
def autoencoder_generator(type_1_data, type_2_data, type_3_data, batch_size):
    
    type_1_data_len = len(type_1_data)
    type_2_data_len = len(type_2_data)
    
    type_3_sampled_for_balance = type_3_data[np.random.choice(len(type_3_data), int((type_1_data_len + type_2_data_len)*1.5),replace=False)]
    type_3_data_len = len(type_3_sampled_for_balance)

    batch_num = int((type_1_data_len + type_2_data_len + type_3_data_len)/batch_size)

    type_1_batch_indexes = GetBatchIndexes(type_1_data_len, batch_num)
    type_2_batch_indexes = GetBatchIndexes(type_2_data_len, batch_num)
    type_3_batch_indexes = GetBatchIndexes(type_3_data_len, batch_num)
    


    counter = 0

    while True:
        input_seg = np.concatenate((type_1_data[type_1_batch_indexes[counter]], type_2_data[type_2_batch_indexes[counter]], type_3_data[type_3_batch_indexes[counter]]))
        X_batch = Segments2Data(input_seg)
        counter += 1
        yield X_batch, X_batch

        if counter >= batch_num:
            counter = 0

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

train_type_1 = np.array(train_segments_set['preictal_ontime'])
train_type_2 = np.array(train_segments_set['ictal'] + train_segments_set['preictal_early'] + train_segments_set['preictal_late'])
train_type_3 = np.array(train_segments_set['postictal'] + train_segments_set['interictal'])

test_type_1 = np.array(test_segments_set['preictal_ontime'])
test_type_2 = np.array(test_segments_set['ictal'] + test_segments_set['preictal_early'] + test_segments_set['preictal_late'])
test_type_3 = np.array(test_segments_set['postictal'] + test_segments_set['interictal'])

fold_n = 5

kf = KFold(n_splits=5, shuffle=True)
epochs = 100
batch_size = 500   # 한번의 gradient update시마다 들어가는 데이터의 사이즈
total_len = len(train_type_1)+len(train_type_2)
total_len = int(total_len*2.5) # 데이터 비율 2:2:6

type_1_kfold_set = kf.split(train_type_1)
type_2_kfold_set = kf.split(train_type_2)
type_3_kfold_set = kf.split(train_type_3)

for _ in range(fold_n):
    encoder_inputs = Input(shape=(21,512,1))
    encoder_outputs = FullChannelEncoder_test(encoded_feature_num=64,inputs = encoder_inputs)
    decoder_outputs = FullChannelDecoder_test(encoder_outputs)
    autoencoder_model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
    autoencoder_model.compile(optimizer = 'Adam', loss='mse',)

    (type_1_train_indexes, type_1_val_indexes) = next(type_1_kfold_set)
    (type_2_train_indexes, type_2_val_indexes) = next(type_2_kfold_set)
    (type_3_train_indexes, type_3_val_indexes) = next(type_3_kfold_set)

    type_1_data_len = len(type_1_train_indexes)
    type_2_data_len = len(type_2_train_indexes)
    type_3_data_len = int((type_1_data_len + type_2_data_len)*1.5)
    train_batch_num = int((type_1_data_len + type_2_data_len + type_3_data_len)/batch_size)

    type_1_data_len = len(type_1_val_indexes)
    type_2_data_len = len(type_2_val_indexes)
    type_3_data_len = int((type_1_data_len + type_2_data_len)*1.5)
    val_batch_num = int((type_1_data_len + type_2_data_len + type_3_data_len)/batch_size)

# %%
if __name__=='__main__':
    history = autoencoder_model.fit_generator(
            autoencoder_generator(train_type_1[type_1_train_indexes], train_type_2[type_2_train_indexes], train_type_3[type_3_train_indexes],batch_size),
            epochs = epochs,
            steps_per_epoch =  train_batch_num,
            validation_data = autoencoder_generator(train_type_1[type_1_val_indexes], train_type_2[type_2_val_indexes], train_type_3[type_3_val_indexes],batch_size),
            validation_steps = val_batch_num,
            workers=8
            )


