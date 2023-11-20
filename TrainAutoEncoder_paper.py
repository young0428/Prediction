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
    
        self.ratio_type_1 = [2,2,2,2]
        self.ratio_type_2 = [2,2,2,2]
        self.ratio_type_3 = [6,6,6,6]
        
        self.batch_size = batch_size
        self.check = True
        self.epoch = 0
        self.cnt = 0
        self.update_period = 5*2
        self.type_1_data = type_1_data
        self.type_2_data = type_2_data
        self.type_3_data = type_3_data
        self.test_on = False
        self.ratio_idx = 0
        self.model_name = model_name
        self.gen_type = gen_type
        self.data_type = data_type

        self.update_data()

    def on_epoch_end(self):
        self.epoch += 1
        self.update_data()
        if self.gen_type == "train":
            if self.epoch % 6 == 0:
                try:
                    test_ae(int(self.epoch/2), 5,2,128, self.model_name)
                except:
                    print("Fail to generate test fig")

    def __len__(self):
        return self.batch_num

    def update_data(self):
        # 데이터 밸런스를 위해 데이터 밸런스 조절 및 resampling
        if self.epoch/self.update_period < 4:
            self.ratio_idx = int(self.epoch/self.update_period)
        else:
            self.ratio_idx = 3
        # ratio에 따라 데이터 갯수 정함
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

    def __getitem__(self, idx):
        input_seg = np.concatenate((self.type_1_data[self.type_1_batch_indexes[idx]], 
                                    self.type_2_sampled[self.type_2_batch_indexes[idx]], 
                                    self.type_3_sampled[self.type_3_batch_indexes[idx]]))
        
        x_batch = Segments2Data(input_seg, self.data_type)
        x_batch = PreProcessing.FilteringSegments(x_batch)

        return x_batch, x_batch

# %%
def train(model_name, type='snu'):
    window_size = 5
    overlap_sliding_size = 2
    normal_sliding_size = window_size
    sr = 128
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
        encoder_outputs = AutoEncoder.FullChannelEncoder_paper_base(inputs = encoder_inputs)
        decoder_outputs = AutoEncoder.FullChannelDecoder_paper_base(encoder_outputs)
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


    train_interval_set = LoadDataset(train_info_file_path)
    train_segments_set = {}

    test_interval_set = LoadDataset(test_info_file_path)
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
                                                            patience=10,
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


