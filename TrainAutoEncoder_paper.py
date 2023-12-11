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
from ModelGenerator import *

tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

# %%
def train(model_name, type='snu', full_ch_model_name = ""):
    window_size = 5
    overlap_sliding_size = 1
    normal_sliding_size = window_size
    sr = 200
    epochs = 100
    batch_size = 500
    state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

    checkpoint_path = f"AutoEncoder/{model_name}/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    full_ch_ae_checkpoint_path = f"AutoEncoder/{full_ch_model_name}/cp.ckpt"
    # for WSL
    if type=='snu':
        train_info_file_path = "/host/d/SNU_DATA/patient_info_snu_train.csv"
        test_info_file_path = "/host/d/SNU_DATA/patient_info_snu_test.csv"
        edf_file_path = "/host/d/SNU_DATA"

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

            
        encoder_inputs = Input(shape=(21,sr*window_size,1))
        encoder_outputs = AutoEncoder.FullChannelEncoder(inputs = encoder_inputs)
        decoder_outputs = AutoEncoder.FullChannelDecoder(encoder_outputs, freq=sr, window_size=window_size)
        model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
        model.compile(optimizer = 'RMSprop', loss='mse')
        if os.path.exists(checkpoint_path):
            print("Model Loaded!")
            model = tf.keras.models.load_model(checkpoint_path)

    elif type=='chb':
        train_info_file_path = "/host/d/CHB/patient_info_chb_train.csv"
        test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
        edf_file_path = "/host/d/CHB"


        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            
        encoder_inputs = Input(shape=(18,sr*window_size,1))
        encoder_outputs = AutoEncoder.FullChannelEncoder_for_CHB(inputs = encoder_inputs)
        decoder_outputs = AutoEncoder.FullChannelDecoder_for_CHB(encoder_outputs)
        model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
        model.compile(optimizer = 'RMSprop', loss='mse')
        if os.path.exists(checkpoint_path):
            print("Model Loaded!")
            model = tf.keras.models.load_model(checkpoint_path)

    elif type=='chb_one_ch':
        train_info_file_path = "/host/d/CHB/patient_info_chb_train.csv"
        test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
        edf_file_path = "/host/d/CHB"

        if os.path.exists(checkpoint_path):
            print("Model Loaded!")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            encoder_inputs = Input(shape=(1,int(sr*window_size),1))
            encoder_outputs = AutoEncoder.OneChannelEncoder(encoder_inputs)
            decoder_outputs = AutoEncoder.OneChannelDecoder(encoder_outputs)
            model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
    elif type=='snu_one_ch':
        train_info_file_path = "/host/d/SNU_DATA/patient_info_snu_train.csv"
        test_info_file_path = "/host/d/SNU_DATA/patient_info_snu_test.csv"
        edf_file_path = "/host/d/SNU_DATA"

        if os.path.exists(checkpoint_path):
            print("Model Loaded!")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            encoder_inputs = Input(shape=(1,int(sr*window_size),1))
            encoder_outputs = AutoEncoder.OneChannelEncoder(encoder_inputs)
            decoder_outputs = AutoEncoder.OneChannelDecoder(encoder_outputs)
            model = Model(inputs=encoder_inputs, outputs=decoder_outputs)

    if not full_ch_model_name == '':
        full_ch_model = tf.keras.models.load_model(full_ch_ae_checkpoint_path)
        encoder_output_layer = get_first_name_like_layer(full_ch_model, 'squeeze')
        encoder_output = encoder_output_layer.output
        full_ch_encoder_model = Model(inputs=encoder_inputs, outputs=encoder_output)
        full_ch_encoder_model.trainable = False
        

    
    model.compile(optimizer = 'Adam', loss='mse')
    model.summary()
    


    train_interval_set,_ = LoadDataset(train_info_file_path)
    train_segments_set = {}

    test_interval_set,_ = LoadDataset(test_info_file_path)
    test_segments_set = {}

    # 상대적으로 데이터 갯수가 적은 것들은 window_size 2초에 sliding_size 1초로 overlap 시켜 데이터 증강
    for state in ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early']:
        train_segments_set[state] = Interval2Segments(train_interval_set[state],edf_file_path, window_size, overlap_sliding_size)
        test_segments_set[state] = Interval2Segments(test_interval_set[state],edf_file_path, window_size, overlap_sliding_size)
        
    for state in ['interictal']:
        train_segments_set[state] = Interval2Segments(train_interval_set[state],edf_file_path, window_size, normal_sliding_size)
        test_segments_set[state] = Interval2Segments(test_interval_set[state],edf_file_path, window_size, normal_sliding_size)

    train_type_1 = np.array(train_segments_set['preictal_ontime']+ train_segments_set['preictal_late'] + train_segments_set['preictal_early'])
    #train_type_2 = np.array(train_segments_set['ictal'] + train_segments_set['preictal_early'] 
    train_type_3 = np.array(train_segments_set['interictal'])

    test_type_1 = np.array(test_segments_set['preictal_ontime']+ test_segments_set['preictal_late'] + test_segments_set['preictal_early'])
    #test_type_2 = np.array(test_segments_set['ictal'] + test_segments_set['preictal_early'] 
    test_type_3 = np.array(test_segments_set['interictal'])

    

    logs = "logs/" + model_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")


    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                    histogram_freq = 1,
                                                    profile_batch = '1,400')
    

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_best_only=True,
                                                    verbose=1)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                            verbose=0,
                                                            patience=5,
                                                            restore_best_weights=True)
    
    backup_callback = tf.keras.callbacks.BackupAndRestore(
      f"./AutoEncoder/{model_name}/training_backup",
      save_freq="epoch",
      delete_checkpoint=True,
    )
    
    train_generator = AutoEncoderGenerator(type_1_data=train_type_1, 
                                            type_3_data=train_type_3,
                                            batch_size=batch_size,
                                            data_type = type,
                                            )
    test_generator = AutoEncoderGenerator(type_1_data=test_type_1, 
                                                type_3_data=test_type_3,
                                                batch_size=batch_size,
                                                data_type = type,
                                            )
    # train_generator = Singal2FullGenerator(type_1_data=train_type_1, 
    #                                         type_3_data=train_type_3,
    #                                         batch_size=batch_size,
    #                                         data_type = type,
    #                                         full_ch_model=full_ch_encoder_model
                                            
    #                                         )
    # test_generator = Singal2FullGenerator(type_1_data=test_type_1, 
    #                                             type_3_data=test_type_3,
    #                                             batch_size=batch_size,
    #                                             data_type = type,
    #                                             full_ch_model=full_ch_encoder_model
    #                                         )
# %%
    history = model.fit(
                train_generator,
                epochs = epochs,
                validation_data = test_generator,
                use_multiprocessing=True,
                workers=32,
                max_queue_size=32,
                shuffle=True,
                callbacks= [ tboard_callback, cp_callback, early_stopping, backup_callback ]
                )
    
    with open(f'./AutoEncoder/{model_name}/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


