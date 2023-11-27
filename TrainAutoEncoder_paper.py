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


import TestFullModel_specific
from readDataset import *
import AutoEncoder 
from LSTMmodel import LSTMLayer
from sklearn.model_selection import KFold
import PreProcessing
from TestAutoEncoder import test_ae

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

# %%
class autoencoder_generator(Sequence):
    def __init__(self,type_1_data, type_2_data, type_3_data, batch_size, model_name, gen_type, data_type):
    
        self.ratio_type_1 = [3,3,3,3]
        self.ratio_type_2 = [3,3,3,3]
        self.ratio_type_3 = [4,4,4,4]
        self.batch_size = batch_size
        self.epoch = 0
        self.update_period = 20
        self.type_1_data = np.array(type_1_data)
        self.type_2_data = np.array(type_2_data)
        self.type_3_data = np.array(type_3_data)
        self.type_1_len = len(type_1_data)
        self.type_2_len = len(type_2_data)
        self.type_3_len = len(type_3_data)
        self.data_type = data_type

        self.iden_mat = np.eye(2)

        self.batch_set, self.batch_num = updateDataSet(self.type_1_len, self.type_2_len, self.type_3_len, [self.ratio_type_1[0], self.ratio_type_2[0], self.ratio_type_3[0]], self.batch_size)

    def on_epoch_end(self):
        self.epoch += 1
        if self.epoch/self.update_period < 4:
            self.ratio_idx = int(self.epoch/self.update_period)
        else:
            self.ratio_idx = 3
        self.ratio_idx = 0
        self.batch_set, self.batch_num = updateDataSet(self.type_1_len, self.type_2_len, self.type_3_len, [self.ratio_type_1[self.ratio_idx], self.ratio_type_2[self.ratio_idx], self.ratio_type_3[self.ratio_idx]], self.batch_size)

    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        x_batch_type_1 = Segments2Data(self.type_1_data[self.batch_set[0][self.batch_set[1][idx]]],self.data_type)
        x_batch_type_2 = Segments2Data(self.type_2_data[self.batch_set[2][self.batch_set[3][idx]]],self.data_type)
        x_batch_type_3 = Segments2Data(self.type_3_data[self.batch_set[4][self.batch_set[5][idx]]],self.data_type)
        x_batch = None
        if x_batch_type_1.ndim == 3:
            if np.all(x_batch== None):
                x_batch = x_batch_type_1
            else:
                x_batch = np.concatenate((x_batch, x_batch_type_1))
            type_1_len = len(x_batch_type_1)
        else:
            type_1_len = 0

        if x_batch_type_2.ndim == 3:
            if np.all(x_batch== None):
                x_batch = x_batch_type_2
            else:
                x_batch = np.concatenate((x_batch, x_batch_type_2))
            type_2_len = len(x_batch_type_2)
        else:
            type_2_len = 0

        if x_batch_type_3.ndim == 3:
            if np.all(x_batch== None):
                x_batch = x_batch_type_3
            else:
                x_batch = np.concatenate((x_batch, x_batch_type_3))
            type_3_len = len(x_batch_type_3)
        else:
            type_3_len = 0

        return x_batch, x_batch

# %%
def train(model_name, type='snu'):
    window_size = 5
    overlap_sliding_size = 2
    normal_sliding_size = window_size
    sr = 256
    check = [True]
    state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

    # for WSL
    if type=='snu':
        train_info_file_path = "/host/d/SNU_DATA/patient_info_snu_train.csv"
        test_info_file_path = "/host/d/SNU_DATA/patient_info_snu_test.csv"
        edf_file_path = "/host/d/SNU_DATA"

        checkpoint_path = f"AutoEncoder/{model_name}/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

            
        encoder_inputs = Input(shape=(21,sr*window_size,1))
        encoder_outputs = AutoEncoder.FullChannelEncoder_for_CHB(inputs = encoder_inputs)
        decoder_outputs = AutoEncoder.FullChannelDecoder_for_CHB(encoder_outputs)
        autoencoder_model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
        autoencoder_model.compile(optimizer = 'RMSprop', loss='mse')
        if os.path.exists(checkpoint_path):
            print("Model Loaded!")
            autoencoder_model = tf.keras.models.load_model(checkpoint_path)

    else:
        train_info_file_path = "/host/d/CHB/patient_info_chb_train.csv"
        test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
        edf_file_path = "/host/d/CHB"

        checkpoint_path = f"AutoEncoder/{model_name}/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

            
        encoder_inputs = Input(shape=(18,sr*window_size,1))
        encoder_outputs = AutoEncoder.FullChannelEncoder_for_CHB(inputs = encoder_inputs)
        decoder_outputs = AutoEncoder.FullChannelDecoder_for_CHB(encoder_outputs)
        autoencoder_model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
        autoencoder_model.compile(optimizer = 'RMSprop', loss='mse')
        if os.path.exists(checkpoint_path):
            print("Model Loaded!")
            autoencoder_model = tf.keras.models.load_model(checkpoint_path)

    
    autoencoder_model.summary()

    ## for window
    # train_info_file_path = "D:/SNU_DATA/patient_info_train.csv"
    # test_info_file_path = "D:/SNU_DATA/patient_info_test.csv"
    # edf_file_path = "D:/SNU_DATA"


    train_interval_set,_ = LoadDataset(train_info_file_path)
    train_segments_set = {}

    test_interval_set,_ = LoadDataset(test_info_file_path)
    test_segments_set = {}

    # 상대적으로 데이터 갯수가 적은 것들은 window_size 2초에 sliding_size 1초로 overlap 시켜 데이터 증강
    for state in ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early']:
        train_segments_set[state] = Interval2Segments(train_interval_set[state],edf_file_path, window_size, overlap_sliding_size)
        
    for state in ['postictal', 'interictal']:
        train_segments_set[state] = Interval2Segments(train_interval_set[state],edf_file_path, window_size, normal_sliding_size)


    # type 1은 True Label데이터 preictal_ontime
    # type 2는 특별히 갯수 맞춰줘야 하는 데이터
    # type 3는 나머지

    # AutoEncoder 단계에서는 1:1:3

    # train_type_1 = np.array(train_segments_set['preictal_ontime'])
    # train_type_2 = np.array(train_segments_set['ictal'] + train_segments_set['preictal_early'] + train_segments_set['preictal_late'])
    # train_type_3 = np.array(train_segments_set['postictal'] + train_segments_set['interictal'])

    train_type_1 = np.array(train_segments_set['preictal_ontime']  )
    train_type_2 = np.array(train_segments_set['ictal'] 
                            + train_segments_set['preictal_early'] 
                            + train_segments_set['preictal_late'] )
    train_type_3 = np.array(train_segments_set['postictal'] + train_segments_set['interictal'])

    fold_n = 5

    kf = KFold(n_splits=5, shuffle=True)
    epochs = 100
    batch_size = 500   # 한번의 gradient update시마다 들어가는 데이터의 사이즈
    total_len = len(train_type_1)+len(train_type_2)
    total_len = int(total_len*2.5) # 데이터 비율 2:2:6

    type_1_kfold_set = kf.split(train_type_1)
    type_2_kfold_set = kf.split(train_type_2)
    type_3_kfold_set = kf.split(train_type_3)



    (type_1_train_indexes, type_1_val_indexes) = next(type_1_kfold_set)
    (type_2_train_indexes, type_2_val_indexes) = next(type_2_kfold_set)
    (type_3_train_indexes, type_3_val_indexes) = next(type_3_kfold_set)
    

    logs = "logs/" + model_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")


    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                    histogram_freq = 1,
                                                    profile_batch = '1,400')
    

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_best_only=True,
                                                    verbose=1)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                            verbose=1,
                                                            patience=5,
                                                            restore_best_weights=True)
    
    backup_callback = tf.keras.callbacks.BackupAndRestore(
      f"./AutoEncoder/{model_name}/training_backup",
      save_freq="epoch",
      delete_checkpoint=True,
    )
    
    train_generator = autoencoder_generator(train_type_1[type_1_train_indexes], 
                                            train_type_2[type_2_train_indexes],
                                            train_type_3[type_3_train_indexes],
                                            batch_size,
                                            model_name,
                                            "train",
                                            type
                                            )
    validation_generator = autoencoder_generator(train_type_1[type_1_val_indexes], 
                                                    train_type_2[type_2_val_indexes],
                                                    train_type_3[type_3_val_indexes],
                                                    batch_size,
                                                    model_name,
                                                    "val",
                                                    type
                                                    )
# %%
    history = autoencoder_model.fit_generator(
                train_generator,
                epochs = epochs,
                validation_data = validation_generator,
                use_multiprocessing=True,
                workers=16,
                shuffle=False,
                callbacks= [ tboard_callback, cp_callback, early_stopping, backup_callback ]
                )
    
    with open(f'./AutoEncoder/{model_name}/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


