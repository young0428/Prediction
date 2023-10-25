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
            # ratio에 따라 데이터 갯수 정함
            self.type_1_sampled_len = len(self.type_1_data)
            self.type_2_sampled_len = min(int((self.type_1_sampled_len/self.ratio_type_1[int(self.epoch/self.update_period)])*self.ratio_type_2[int(self.epoch/self.update_period)]),len(self.type_2_data))
            self.type_3_sampled_len = int((self.type_1_sampled_len/self.ratio_type_1[int(self.epoch/self.update_period)])*self.ratio_type_3[int(self.epoch/self.update_period)])
            # Sampling mask 생성
            self.type_2_sampling_mask = sorted(np.random.choice(len(self.type_2_data), self.type_2_sampled_len, replace=False))
            self.type_3_sampling_mask = sorted(np.random.choice(len(self.type_3_data), self.type_3_sampled_len, replace=False))

            self.type_2_sampled = self.type_2_data[self.type_2_sampling_mask]
            self.type_3_sampled = self.type_3_data[self.type_3_sampling_mask]

            self.batch_num = int((self.type_1_sampled_len + self.type_2_sampled_len + self.type_3_sampled_len)/self.batch_size)
            
            self.type_1_batch_indexes = GetBatchIndexes(self.type_1_sampled_len, self.batch_num)
            self.type_2_batch_indexes = GetBatchIndexes(self.type_2_sampled_len, self.batch_num)
            self.type_3_batch_indexes = GetBatchIndexes(self.type_3_sampled_len, self.batch_num)
    
    def __len__(self):
        return self.batch_num
    
    def __getitem__(self, idx):
        input_seg = np.concatenate((self.type_1_data[self.type_1_batch_indexes[idx]], 
                                    self.type_2_sampled[self.type_2_batch_indexes[idx]], 
                                    self.type_3_sampled[self.type_3_batch_indexes[idx]]))
        y_batch = np.concatenate( ( np.ones(len(self.type_1_batch_indexes[idx])), 
                                   (np.zeros(len(self.type_2_batch_indexes[idx]))), 
                                   (np.zeros(len(self.type_3_batch_indexes[idx]))) )  )
        
        y_batch = y_batch.tolist()
        y_batch = list(map(int,y_batch))
        y_batch = np.eye(2)[y_batch]
        
        data = Segments2Data(input_seg) # (batch, eeg_channel, data)
        x_batch = np.split(data, 10, axis=-1) # (10, batch, eeg_channel, data)
        x_batch = np.transpose(x_batch,(1,0,2,3))

        if (idx+1) % int(self.batch_num / 5) == 0:
            self.type_3_sampling_mask = sorted(np.random.choice(len(self.type_3_data), self.type_3_sampled_len, replace=False))
            self.type_3_sampled = self.type_3_data[self.type_3_sampling_mask]
            self.type_3_batch_indexes = GetBatchIndexes(self.type_3_sampled_len, self.batch_num)

        return x_batch, y_batch

# %%
if __name__=='__main__':
    window_size = 20
    overlap_sliding_size = 10
    normal_sliding_size = 20
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

    autoencoder_model_path = "AutoEncoder_training_0/cp.ckpt"

    encoder_inputs = Input(shape=(21,512,1))
    encoder_outputs = FullChannelEncoder(encoded_feature_num=512,inputs = encoder_inputs)
    decoder_outputs = FullChannelDecoder(encoder_outputs)
    autoencoder_model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
    autoencoder_model.load_weights(autoencoder_model_path)

    encoder_input = autoencoder_model.input
    encoder_output = autoencoder_model.get_layer("dense").output
    encoder_model = Model(inputs=encoder_input, outputs=encoder_output)
    encoder_model.trainable = False

    fullmodel_input = Input(shape=(10,21,512,1))
    ts_output = TimeDistributed(encoder_model)(fullmodel_input)
    lstm_output = LSTMLayer(ts_output)
    full_model = Model(inputs=fullmodel_input, outputs=lstm_output)

    for _ in range(fold_n):
        (type_1_train_indexes, type_1_val_indexes) = next(type_1_kfold_set)
        (type_2_train_indexes, type_2_val_indexes) = next(type_2_kfold_set)
        (type_3_train_indexes, type_3_val_indexes) = next(type_3_kfold_set)
        checkpoint_path = f"FullModel_training_{_}/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
    
        if os.path.exists(f"./FullModel_training_{_+1}"):
            continue
        else:
            fullmodel_input = Input(shape=(10,21,512,1))
            ts_output = TimeDistributed(encoder_model)(fullmodel_input)
            lstm_output = LSTMLayer(ts_output)
            full_model = Model(inputs=fullmodel_input, outputs=lstm_output)
            full_model.compile(optimizer = 'Adam',metrics=['acc'] ,loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.2))

            if os.path.exists(f"./FullModel_training_{_}"):
                print("Model Loaded!")
                full_model = tf.keras.models.load_model(checkpoint_path)
        full_model.summary()

        type_1_data_len = len(type_1_train_indexes)
        type_2_data_len = len(type_2_train_indexes)
        type_3_data_len = int((type_1_data_len + type_2_data_len)*1.5)
        train_batch_num = int((type_1_data_len + type_2_data_len + type_3_data_len)/batch_size)

        type_1_data_len = len(type_1_val_indexes)
        type_2_data_len = len(type_2_val_indexes)
        type_3_data_len = int((type_1_data_len + type_2_data_len)*1.5)
        val_batch_num = int((type_1_data_len + type_2_data_len + type_3_data_len)/batch_size)

        type_1_data_len = len(test_type_1)
        type_2_data_len = len(test_type_2)
        type_3_data_len = len(test_type_3)
        test_batch_num = int((type_1_data_len + type_2_data_len + type_3_data_len)/batch_size)

        logs = "logs/lstm_512"

        

        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                        histogram_freq = 1,
                                                        profile_batch = '100,200')
        

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
                    steps_per_epoch =  train_batch_num,
                    validation_data = test_generator,
                    validation_steps = test_batch_num,
                    use_multiprocessing=True,
                    workers=24,
                    callbacks= [ tboard_callback, cp_callback ]
                    )
        
        with open(f'./FullModel_training_{_}/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)


