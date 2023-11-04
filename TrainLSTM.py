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
from tensorflow.keras.layers import Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import F1Score, Recall, Precision


from readDataset import LoadDataset, Interval2Segments, Segments2Data
from AutoEncoder import FullChannelEncoder, FullChannelDecoder
from LSTMmodel import LSTMLayer
from sklearn.model_selection import KFold
import PreProcessing



gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

# %%
class FullModel_generator(Sequence):
    def __init__(self,type_1_data, type_2_data, type_3_data, batch_size):
        
        self.ratio_type_1 = [5,4,3,2]
        self.ratio_type_2 = [3,3,3,2]
        self.ratio_type_3 = [2,3,4,6]
        self.batch_size = batch_size
        self.epoch = 0
        self.update_period = 20
        self.type_1_data = type_1_data
        self.type_2_data = type_2_data
        self.type_3_data = type_3_data


        self.update_data()

    def on_epoch_end(self):
        self.epoch += 1
        self.update_data()

    def update_data(self):
        # 데이터 밸런스를 위해 데이터 밸런스 조절 및 resampling
        if self.epoch/self.update_period < 4:
            self.ratio_idx = int(self.epoch/self.update_period)
        else:
            self.ratio_idx = 3
        # ratio에 따라 데이터 갯수 정함
        self.ratio_idx = 0
        self.type_1_sampled_len = len(self.type_1_data)
        self.type_2_sampled_len = min(int((self.type_1_sampled_len/self.ratio_type_1[self.ratio_idx])*self.ratio_type_2[self.ratio_idx]),len(self.type_2_data))
        self.type_3_sampled_len = int((self.type_1_sampled_len/self.ratio_type_1[self.ratio_idx])*self.ratio_type_3[self.ratio_idx])
        # Sampling mask 생성
        self.type_2_sampling_mask = sorted(np.random.choice(len(self.type_2_data), self.type_2_sampled_len, replace=False))
        self.type_3_sampling_mask = sorted(np.random.choice(len(self.type_3_data), self.type_3_sampled_len, replace=False))

        self.type_2_sampled = self.type_2_data[self.type_2_sampling_mask]
        self.type_3_sampled = self.type_3_data[self.type_3_sampling_mask]

        self.batch_num = int((self.type_1_sampled_len + self.type_2_sampled_len + self.type_3_sampled_len)/self.batch_size)
        
        self.type_1_batch_indexes = PreProcessing.GetBatchIndexes(self.type_1_sampled_len, self.batch_num)
        self.type_2_batch_indexes = PreProcessing.GetBatchIndexes(self.type_2_sampled_len, self.batch_num)
        self.type_3_batch_indexes = PreProcessing.GetBatchIndexes(self.type_3_sampled_len, self.batch_num)
    
    def __len__(self):
        return self.batch_num
    
    def __getitem__(self, idx):
        input_seg = np.concatenate((self.type_1_data[self.type_1_batch_indexes[idx]], 
                                    self.type_2_sampled[self.type_2_batch_indexes[idx]], 
                                    self.type_3_sampled[self.type_3_batch_indexes[idx]]))
        y_batch = np.concatenate( ( np.ones(len(self.type_1_batch_indexes[idx])), 
                                   (np.zeros(len(self.type_2_batch_indexes[idx]))), 
                                   (np.zeros(len(self.type_3_batch_indexes[idx]))) )  )
        
        x_batch = Segments2Data(input_seg) # (batch, eeg_channel, data)

        if (idx+1) % int(self.batch_num / 3) == 0:
            self.type_3_sampling_mask = sorted(np.random.choice(len(self.type_3_data), self.type_3_sampled_len, replace=False))
            self.type_3_sampled = self.type_3_data[self.type_3_sampling_mask]
            self.type_3_batch_indexes = PreProcessing.GetBatchIndexes(self.type_3_sampled_len, self.batch_num)

        return x_batch, y_batch

# %%
def train(model_name, encoder_model_name):
    window_size = 5
    overlap_sliding_size = 2
    normal_sliding_size = window_size
    sr = 128
    state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

    # for WSL
    train_info_file_path = "/host/d/SNU_DATA/SNU_patient_info_train.csv"
    test_info_file_path = "/host/d/SNU_DATA/SNU_patient_info_test.csv"
    edf_file_path = "/host/d/SNU_DATA"

    # #for window
    # train_info_file_path = "D:/SNU_DATA/SNU_patient_info_train.csv"
    # test_info_file_path = "D:/SNU_DATA/SNU_patient_info_test.csv"
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
     # 데이터 비율 2:2:6

    type_1_kfold_set = kf.split(train_type_1)
    type_2_kfold_set = kf.split(train_type_2)
    type_3_kfold_set = kf.split(train_type_3)

    autoencoder_model_path = f"AutoEncoder/{encoder_model_name}/cp.ckpt"

    encoder_inputs = Input(shape=(21,int(sr*window_size),1))
    encoder_outputs = FullChannelEncoder(inputs = encoder_inputs)
    decoder_outputs = FullChannelDecoder(encoder_outputs, window_size=window_size, freq=sr)
    autoencoder_model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
    autoencoder_model.load_weights(autoencoder_model_path)

    encoder_output = autoencoder_model.get_layer("tf.compat.v1.squeeze_2").output
    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_output)
    encoder_model.trainable = False

    ## 마지막 Conv layer Find Tuning
    # encoder_layer_for_fine_tuning = ['conv2d_3','conv2d_7','conv2d_11','conv2d_15','conv2d_19','conv2d_23']
    # for layer_name_for_find_tune in encoder_layer_for_fine_tuning:
    #     autoencoder_model.get_layer(layer_name_for_find_tune).trainable=True

    lstm_output = LSTMLayer(encoder_output)
    full_model = Model(inputs=encoder_inputs, outputs=lstm_output)

    (type_1_train_indexes, type_1_val_indexes) = next(type_1_kfold_set)
    (type_2_train_indexes, type_2_val_indexes) = next(type_2_kfold_set)
    (type_3_train_indexes, type_3_val_indexes) = next(type_3_kfold_set)
    checkpoint_path = f"LSTM/{model_name}/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    full_model.compile(optimizer = 'RMSprop',
                        metrics=[
                                tf.keras.metrics.BinaryAccuracy(threshold=0), 
                                tf.keras.metrics.Recall(thresholds=0), 
                                tf.keras.metrics.Precision(thresholds=0),
                                ] ,
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1))

    if os.path.exists(f"./LSTM/{model_name}"):
        print("Model Loaded!")
        full_model = tf.keras.models.load_model(checkpoint_path)

    logs = f"logs/{model_name}"    

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                    histogram_freq = 1,
                                                    profile_batch = '100,200')
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_precision', 
                                                        verbose=1,
                                                        patience=10,
                                                        mode='max',
                                                        restore_best_weights=True)
    

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_best_only=True,
                                                    verbose=1)
    
    train_generator = FullModel_generator(train_type_1[type_1_train_indexes], train_type_2[type_2_train_indexes], train_type_3[type_3_train_indexes],batch_size)
    validation_generator = FullModel_generator(train_type_1[type_1_val_indexes], train_type_2[type_2_val_indexes], train_type_3[type_3_val_indexes],batch_size)
    test_generator = FullModel_generator(test_type_1, test_type_2, test_type_3, batch_size)
# %%
    history = full_model.fit_generator(
                train_generator,
                epochs = epochs,
                validation_data = test_generator,
                use_multiprocessing=True,
                workers=16,
                callbacks= [ tboard_callback, cp_callback, early_stopping ]
                )
    
    with open(f'./LSTM/{model_name}/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


