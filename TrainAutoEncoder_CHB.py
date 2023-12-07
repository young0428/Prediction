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
def train(model_name):
    window_size = 5
    overlap_sliding_size = 2
    normal_sliding_size = window_size
    ch_num = 18
    sr = 200
    state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

    # for WSL
    train_info_file_path = "/host/d/CHB/patient_info_chb_train.csv"
    test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
    edf_file_path = "/host/d/CHB"

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

    train_type_1 = np.array(train_segments_set['preictal_ontime'] + train_segments_set['preictal_late'] )
    train_type_2 = np.array(train_segments_set['ictal'] 
                            + train_segments_set['preictal_early'] 
                            )
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
    checkpoint_path = f"AutoEncoder/{model_name}/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

        
    encoder_inputs = Input(shape=(ch_num,sr*window_size,1))
    encoder_outputs = AutoEncoder.FullChannelEncoder_for_CHB(inputs = encoder_inputs)
    decoder_outputs = AutoEncoder.FullChannelDecoder_for_CHB(encoder_outputs)
    autoencoder_model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
    autoencoder_model.compile(optimizer = 'RMSprop', loss='mse')
    if os.path.exists(checkpoint_path):
        print("Model Loaded!")
        autoencoder_model = tf.keras.models.load_model(checkpoint_path)

    autoencoder_model.summary()

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
                                            "train"
                                            
                                            )
    validation_generator = autoencoder_generator(train_type_1[type_1_val_indexes], 
                                                    train_type_2[type_2_val_indexes],
                                                    train_type_3[type_3_val_indexes],
                                                    batch_size,
                                                    model_name,
                                                    "val"
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


