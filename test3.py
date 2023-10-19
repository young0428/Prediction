# %%
import os

os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from datetime import datetime
import sys
import numpy as np
import random
import operator
import matplotlib as plt
import pickle

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence


from readDataset import LoadDataset, Interval2Segments, Segments2Data
from AutoEncoder import FullChannelEncoder, FullChannelDecoder
from LSTMmodel import LSTMLayer
from sklearn.model_selection import KFold
from PreProcessing import GetBatchIndexes


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

# %%
class autoencoder_generator(Sequence):
    def __init__(self,type_1_data, type_2_data, type_3_data, batch_size):
        
        self.type_1_data = type_1_data
        self.type_2_data = type_2_data
        self.type_3_data = type_3_data

        self.type_1_data_len = len(type_1_data)
        self.type_2_data_len = len(type_2_data)
        
        type_3_sampled_for_balance = type_3_data[np.random.choice(len(type_3_data), int((self.type_1_data_len + self.type_2_data_len)*1.5),replace=False)]
        self.type_3_data_len = len(type_3_sampled_for_balance)

        self.batch_num = int((self.type_1_data_len + self.type_2_data_len + self.type_3_data_len)/batch_size)

        self.type_1_batch_indexes = GetBatchIndexes(self.type_1_data_len, self.batch_num)
        self.type_2_batch_indexes = GetBatchIndexes(self.type_2_data_len, self.batch_num)
        self.type_3_batch_indexes = GetBatchIndexes(self.type_3_data_len, self.batch_num)

    def __len__(self):
        return self.batch_num
    
    def __getitem__(self, idx):
        input_seg = np.concatenate((self.type_1_data[self.type_1_batch_indexes[idx]], self.type_2_data[self.type_2_batch_indexes[idx]], self.type_3_data[self.type_3_batch_indexes[idx]]))
        X_batch = Segments2Data(input_seg)
        #X_batch = np.random.standard_normal((300,21,512))
        return X_batch, X_batch

# %%
if __name__=='__main__':
    window_size = 2
    overlap_sliding_size = 1
    normal_sliding_size = window_size
    state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

    # for WSL
    train_info_file_path = "/host/d/SNU_DATA/patient_info_train.csv"
    test_info_file_path = "/host/d/SNU_DATA/patient_info_test.csv"
    edf_file_path = "/host/d/SNU_DATA"

    ## for window
    # train_info_file_path = "D:/SNU_DATA/patient_info_train.csv"
    # test_info_file_path = "D:/SNU_DATA/patient_info_test.csv"
    # edf_file_path = "D:/SNU_DATA"


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
        (type_1_train_indexes, type_1_val_indexes) = next(type_1_kfold_set)
        (type_2_train_indexes, type_2_val_indexes) = next(type_2_kfold_set)
        (type_3_train_indexes, type_3_val_indexes) = next(type_3_kfold_set)
        checkpoint_path = f"AutoEncoder_training_{_}/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        if os.path.exists(f"./AutoEncoder_training_{_+1}"):
            continue
        else:
            if os.path.exists(f"./AutoEncoder_training_{_}"):
                autoencoder_model = tf.keras.models.load_model(checkpoint_path)
            else:
                encoder_inputs = Input(shape=(21,512,1))
                encoder_outputs = FullChannelEncoder(encoded_feature_num=64,inputs = encoder_inputs)
                decoder_outputs = FullChannelDecoder(encoder_outputs)
                autoencoder_model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
                autoencoder_model.compile(optimizer = 'Adam', loss='mse',)
                
            

        type_1_data_len = len(type_1_train_indexes)
        type_2_data_len = len(type_2_train_indexes)
        type_3_data_len = int((type_1_data_len + type_2_data_len)*1.5)
        train_batch_num = int((type_1_data_len + type_2_data_len + type_3_data_len)/batch_size)

        type_1_data_len = len(type_1_val_indexes)
        type_2_data_len = len(type_2_val_indexes)
        type_3_data_len = int((type_1_data_len + type_2_data_len)*1.5)
        val_batch_num = int((type_1_data_len + type_2_data_len + type_3_data_len)/batch_size)
        logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        

        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                        histogram_freq = 1,
                                                        profile_batch = '1,20')
        

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_best_only=True,
                                                        verbose=1)
        
        train_generator = autoencoder_generator(train_type_1[type_1_train_indexes], train_type_2[type_2_train_indexes], train_type_3[type_3_train_indexes],batch_size)
        validation_generator = autoencoder_generator(train_type_1[type_1_val_indexes], train_type_2[type_2_val_indexes], train_type_3[type_3_val_indexes],batch_size)
# %%
        history = autoencoder_model.fit_generator(
                    train_generator,
                    epochs = epochs,
                    steps_per_epoch =  train_batch_num,
                    validation_data = validation_generator,
                    validation_steps = val_batch_num,
                    use_multiprocessing=True,
                    workers=6,
                    callbacks= [ tboard_callback, cp_callback ]
                    )
        
        with open(f'./AutoEncoder_training_{_}/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)


