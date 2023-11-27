
# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE']='gpu_private'
from datetime import datetime
import sys
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import pickle
import copy
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import F1Score, Recall, Precision

import TestFullModel_specific
from readDataset import *
import AutoEncoder 
from LSTMmodel import LSTMLayer
from sklearn.model_selection import KFold
import PreProcessing


tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)




# %%
class FullModel_generator(Sequence):
    def __init__(self,type_1_data, type_2_data, type_3_data, batch_size, data_type = 'snu', gen_type = 'train'):
        
        self.ratio_type_1 = [5]*4
        self.ratio_type_2 = [1]*4
        self.ratio_type_3 = [5]*4
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
        if gen_type == 'val':
            self.batch_size = 50
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
        batch_concat = np.concatenate((self.type_1_data[self.batch_set[0][self.batch_set[1][idx]]],
                                       self.type_2_data[self.batch_set[2][self.batch_set[3][idx]]],
                                       self.type_3_data[self.batch_set[4][self.batch_set[5][idx]]]))
        x_batch = Segments2Data(batch_concat, self.data_type)
        type_1_len = len(self.type_1_data[self.batch_set[0][self.batch_set[1][idx]]])
        type_2_len = len(self.type_2_data[self.batch_set[2][self.batch_set[3][idx]]])
        type_3_len = len(self.type_3_data[self.batch_set[4][self.batch_set[5][idx]]])
        # x_batch_type_1 = Segments2Data(self.type_1_data[self.batch_set[0][self.batch_set[1][idx]]],self.data_type)
        # x_batch_type_2 = Segments2Data(self.type_2_data[self.batch_set[2][self.batch_set[3][idx]]],self.data_type)
        # x_batch_type_3 = Segments2Data(self.type_3_data[self.batch_set[4][self.batch_set[5][idx]]],self.data_type)
        # x_batch = None
        # if x_batch_type_1.ndim == 3:
        #     if np.all(x_batch== None):
        #         x_batch = x_batch_type_1
        #     else:
        #         x_batch = np.concatenate((x_batch, x_batch_type_1))
        #     type_1_len = len(x_batch_type_1)
        # else:
        #     type_1_len = 0

        # if x_batch_type_2.ndim == 3:
        #     if np.all(x_batch== None):
        #         x_batch = x_batch_type_2
        #     else:
        #         x_batch = np.concatenate((x_batch, x_batch_type_2))
        #     type_2_len = len(x_batch_type_2)
        # else:
        #     type_2_len = 0

        # if x_batch_type_3.ndim == 3:
        #     if np.all(x_batch== None):
        #         x_batch = x_batch_type_3
        #     else:
        #         x_batch = np.concatenate((x_batch, x_batch_type_3))
        #     type_3_len = len(x_batch_type_3)
        # else:
        #     type_3_len = 0

        #x_batch = PreProcessing.FilteringSegments(x_batch)
        y_batch = np.concatenate(( np.ones(type_1_len)*1, 
                                   np.ones(type_2_len)*0, 
                                   np.ones(type_3_len)*0))
        
        #y_batch = self.iden_mat[y_categorical]

        return x_batch, y_batch

def train(model_name, encoder_model_name, data_type = 'snu'):
    window_size = 5
    overlap_sliding_size = 5
    normal_sliding_size = window_size
    sr = 256
    epochs = 100
    batch_size = 500
    state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

    # for WSL
    autoencoder_model_path = f"AutoEncoder/{encoder_model_name}/cp.ckpt"
    if data_type=='snu':
        # train_info_file_path = "/host/d/SNU_DATA/patient_info_snu_train.csv"
        # test_info_file_path = "/host/d/SNU_DATA/patient_info_snu_test.csv"
        # edf_file_path = "/host/d/SNU_DATA"

        train_info_file_path = "/home/SNU_DATA/patient_info_snu_train.csv"
        test_info_file_path = "/home/SNU_DATA/patient_info_snu_test.csv"
        edf_file_path = "/home/SNU_DATA"

        checkpoint_path = f"AutoEncoder/{encoder_model_name}/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

            
        encoder_inputs = Input(shape=(21,int(sr*window_size),1))
        encoder_outputs = AutoEncoder.FullChannelEncoder_paper_base(inputs = encoder_inputs)
        decoder_outputs = AutoEncoder.FullChannelDecoder_paper_base(encoder_outputs)
        autoencoder_model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
        autoencoder_model.load_weights(autoencoder_model_path)

        encoder_output_layer = get_first_name_like_layer(autoencoder_model, 'squeeze')
        encoder_output = encoder_output_layer.output
        encoder_model = Model(inputs=encoder_inputs, outputs=encoder_output)
        encoder_model.trainable = False

    

    else:
        # train_info_file_path = "/host/d/CHB/patient_info_chb_train.csv"
        # test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
        # edf_file_path = "/host/d/CHB"

        train_info_file_path = "/home/CHB/patient_info_chb_train.csv"
        test_info_file_path = "/home/CHB/patient_info_chb_test.csv"
        edf_file_path = "/home/CHB"

        checkpoint_path = f"AutoEncoder/{encoder_model_name}/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

            
        encoder_inputs = Input(shape=(18,int(sr*window_size),1))
        encoder_outputs = AutoEncoder.FullChannelEncoder_for_CHB(inputs = encoder_inputs)
        decoder_outputs = AutoEncoder.FullChannelDecoder_for_CHB(encoder_outputs)
        autoencoder_model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
        autoencoder_model.load_weights(autoencoder_model_path)

        encoder_output_layer = get_first_name_like_layer(autoencoder_model, 'squeeze')
        encoder_output = encoder_output_layer.output
        encoder_model = Model(inputs=encoder_inputs, outputs=encoder_output)
        encoder_model.trainable = False


    train_interval_set, train_interval_overall = LoadDataset(train_info_file_path)
    train_segments_set = {}
    val_segments_set = {}

    # %%
    channel_filtered_intervals = FilteringByChannel(train_interval_overall, edf_file_path, 'chb')
    interval_dict_key_patient_name = Interval2NameKeyDict(channel_filtered_intervals)
    filtered_interval_dict, ictal_num = FilterValidatePatient(interval_dict_key_patient_name)
    for patient_name in filtered_interval_dict.keys() :
        
        train_val_sets = MakeValidationIntervalSet(filtered_interval_dict[patient_name])

        patient_sens_sum = 0
        patient_fpr_sum = 0
        for idx, set in enumerate(train_val_sets):
            for i in range(len(set['train'])):
                set['train'][i][1] = int(set['train'][i][1])
                set['train'][i][2] = int(set['train'][i][2])

            for i in range(len(set['val'])):
                set['val'][i][1] = int(set['val'][i][1])
                set['val'][i][2] = int(set['val'][i][2])

            print(set['val'])
            train_intervals = IntervalList2Dict(set['train'])
            val_intervals = IntervalList2Dict(set['val'])
        # 상대적으로 데이터 갯수가 적은 것들은 window_size 2초에 sliding_size 1초로 overlap 시켜 데이터 증강
            for state in ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early']:
                if state in train_intervals.keys():
                    train_segments_set[state] = Interval2Segments(train_intervals[state],edf_file_path, window_size, overlap_sliding_size)
                else:
                    train_segments_set[state] = []

                if state in val_intervals.keys():
                    val_segments_set[state] = Interval2Segments(val_intervals[state],edf_file_path, window_size, overlap_sliding_size)
                else:
                    val_segments_set[state] = []
                
            for state in ['postictal', 'interictal']:
                if state in train_intervals.keys():
                    train_segments_set[state] = Interval2Segments(train_intervals[state],edf_file_path, window_size, normal_sliding_size)
                else:
                    train_segments_set[state] = []
                    
                if state in val_intervals.keys():
                    val_segments_set[state] = Interval2Segments(val_intervals[state],edf_file_path, window_size, normal_sliding_size)
                else:
                    val_segments_set[state] = []

            train_type_1 = np.array(train_segments_set['preictal_ontime']+ train_segments_set['preictal_late'] )
            train_type_2 = np.array(train_segments_set['ictal'] + train_segments_set['preictal_early'])
            train_type_3 = np.array(train_segments_set['postictal'] + train_segments_set['interictal'])

            val_type_1 = np.array(val_segments_set['preictal_ontime']+ val_segments_set['preictal_late'] )
            val_type_2 = np.array(val_segments_set['ictal'] + val_segments_set['preictal_early'] )
            val_type_3 = np.array(val_segments_set['postictal'] + val_segments_set['interictal'])
            
            lstm_output = LSTMLayer(encoder_output, 10)
            full_model = Model(inputs=encoder_inputs, outputs=lstm_output)

            checkpoint_path = f"./LSTM/{model_name}/{patient_name}/set{idx+1}/cp.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                with open(f'{checkpoint_dir}/training_done','w') as f:
                    f.write('0')
            else:
                with open(f'{checkpoint_dir}/training_done','r') as f:
                    line = f.readline()
                    if line == '1':
                        print(f"{patient_name}, set{idx+1} training already done!!")
                        with open(f'./LSTM/{model_name}/{patient_name}/set{idx+1}/ValResults', 'rb') as file_pi:
                            result_list = pickle.load(file_pi)
                            print(f'set{idx+1} Sensitivity : {result_list[2]}   FPR : {result_list[3]}')
                            patient_sens_sum += result_list[2]
                            patient_fpr_sum += result_list[3]
                        continue

            full_model.compile(optimizer = 'RMSprop',
                                metrics=[
                                        # tf.keras.metrics.BinaryAccuracy(threshold=0),
                                        # tf.keras.metrics.Recall(thresholds=0)
                                        tf.keras.metrics.BinaryAccuracy(threshold=0.5),
                                        tf.keras.metrics.Recall(thresholds=0.5)
                                        ] ,
                                loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) )

            if os.path.exists(checkpoint_path):
                print("Model Loaded!")
                full_model = tf.keras.models.load_model(checkpoint_path)

            logs = f"logs/{model_name}/{patient_name}/set{idx+1}"    

            tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                            histogram_freq = 1,
                                                            profile_batch = '100,200')
            
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', 
                                                                verbose=0,
                                                                patience=5,
                                                                restore_best_weights=True)
            
            backup_callback = tf.keras.callbacks.BackupAndRestore(
            f"{checkpoint_dir}/training_backup",
            save_freq="epoch",
            delete_checkpoint=True,
            )
            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                            save_best_only=True,
                                                            verbose=0)
            
            # %%
            
            train_generator = FullModel_generator(train_type_1, train_type_2, train_type_3,batch_size,data_type, 'train')
            validation_generator = FullModel_generator(val_type_1, val_type_2, val_type_3,batch_size,data_type,'val')

            history = full_model.fit(
                        train_generator,
                        epochs = epochs,
                        validation_data = validation_generator,
                        use_multiprocessing=True,
                        workers=8,
                        callbacks= [  cp_callback, early_stopping, backup_callback ]
            )

            matrix, postprocessed_matrix, sens, fpr, seg_results = TestFullModel_specific.validation(checkpoint_path,val_intervals, data_type, 5,4)
            patient_sens_sum += sens
            patient_fpr_sum += fpr
            result_list = [matrix, postprocessed_matrix, sens, fpr, seg_results]
            print(f'{patient_name} set{idx+1},  Sensitivity : {sens}    FPR : {fpr}/h')
            print(f'{patient_name} Avg,     Senstivity : {patient_sens_sum/(idx+1)} FPR : {patient_fpr_sum/(idx+1)}/h')
            
            

            with open(f'{checkpoint_dir}/training_done','w') as f:
                f.write('1')
            with open(f'./LSTM/{model_name}/{patient_name}/set{idx+1}/trainHistoryDict', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            with open(f'./LSTM/{model_name}/{patient_name}/set{idx+1}/ValResults', 'wb') as file_pi:
                pickle.dump(result_list, file_pi)

            
            SaveAsHeatmap(matrix, f"{checkpoint_dir}/categorical_matrix.png")
            plt.clf()
            SaveAsHeatmap(postprocessed_matrix, f"{checkpoint_dir}/tf_matrix.png" )   
            plt.clf()

            del full_model

        patient_sens_sum = 0
        patient_fpr_sum = 0

def SaveAsHeatmap(matrix, path):
    sns.heatmap(matrix,annot=True, cmap='Blues')
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.savefig(path)
    plt.clf()




