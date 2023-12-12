
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
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
from ModelGenerator import *
from dilationmodel import *
from vit_tensorflow.mobile_vit import one_channel_mobile_vit


tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

# %%
def train(model_name, encoder_model_name, data_type = 'snu'):
    window_size = 120
    overlap_sliding_size = 10
    normal_sliding_size = window_size
    sr = 200
    epochs = 100
    batch_size = 50
    state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

    # for WSL
    autoencoder_model_path = f"AutoEncoder/{encoder_model_name}/cp.ckpt"
    if data_type=='snu':
        # train_info_file_path = "/host/d/SNU_DATA/patient_info_snu_train.csv"
        # test_info_file_path = "/host/d/SNU_DATA/patient_info_snu_test.csv"
        # edf_file_path = "/host/d/SNU_DATA"

        train_info_file_path = "/host/d/SNU_DATA/patient_info_snu_train.csv"
        test_info_file_path = "/host/d/SNU_DATA/patient_info_snu_test.csv"
        edf_file_path = "/host/d/SNU_DATA"

        checkpoint_path = f"AutoEncoder/{encoder_model_name}/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

            
        encoder_inputs = Input(shape=(21,int(sr*window_size),1))
        encoder_outputs = AutoEncoder.FullChannelEncoder(inputs = encoder_inputs)
        decoder_outputs = AutoEncoder.FullChannelDecoder(encoder_outputs, freq=sr, window_size=window_size)
        autoencoder_model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
        autoencoder_model.load_weights(autoencoder_model_path)

        encoder_output_layer = get_first_name_like_layer(autoencoder_model, 'squeeze')
        encoder_output = encoder_output_layer.output
        encoder_model = Model(inputs=encoder_inputs, outputs=encoder_output)
        encoder_model.trainable = False
    elif data_type == 'chb':
        # train_info_file_path = "/host/d/CHB/patient_info_chb_train.csv"
        # test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
        # edf_file_path = "/host/d/CHB"

        train_info_file_path = "/host/d/CHB/patient_info_chb_train.csv"
        test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
        edf_file_path = "/host/d/CHB"

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
    elif data_type=='chb_one_ch':
        train_info_file_path = "/host/d/CHB/patient_info_chb_train.csv"
        test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
        edf_file_path = "/home/CHB"

        

    elif data_type=='snu_one_ch':
        train_info_file_path = "/host/d/SNU_DATA/patient_info_snu_train.csv"
        test_info_file_path = "/host/d/SNU_DATA/patient_info_snu_test.csv"
        edf_file_path = "/host/d/SNU_DATA"

        aemodel = tf.keras.models.load_model(autoencoder_model_path)

        encoder_inputs = aemodel.input
        encoder_last_layer = get_first_name_like_layer(aemodel, 'squeeze')
        encoder_output = encoder_last_layer.output
        encoder_model = Model(inputs = encoder_inputs, outputs = encoder_output)
        encoder_model.trainable = False




    train_interval_set, train_interval_overall = LoadDataset(train_info_file_path)
    train_segments_set = {}
    val_segments_set = {}

    # %%
    channel_filtered_intervals = FilteringByChannel(train_interval_overall, edf_file_path, data_type)
    interval_dict_key_patient_name = Interval2NameKeyDict(channel_filtered_intervals)
    filtered_interval_dict, ictal_num = FilterValidatePatient(interval_dict_key_patient_name)
    total_sens_sum = 0
    total_fpr_sum = 0
    for patient_idx, patient_name in enumerate(filtered_interval_dict.keys()) :
        
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
            for state in ['preictal_ontime']:
                if state in train_intervals.keys():
                    train_segments_set[state] = Interval2Segments(train_intervals[state],edf_file_path, window_size, overlap_sliding_size)
                else:
                    train_segments_set[state] = []

                if state in val_intervals.keys():
                    val_segments_set[state] = Interval2Segments(val_intervals[state],edf_file_path, window_size, overlap_sliding_size)
                else:
                    val_segments_set[state] = []
                
            for state in ['interictal']:
                if state in train_intervals.keys():
                    train_segments_set[state] = Interval2Segments(train_intervals[state],edf_file_path, window_size, normal_sliding_size)
                else:
                    train_segments_set[state] = []
                    
                if state in val_intervals.keys():
                    val_segments_set[state] = Interval2Segments(val_intervals[state],edf_file_path, window_size, normal_sliding_size)
                else:
                    val_segments_set[state] = []

            train_type_1 = np.array(train_segments_set['preictal_ontime'])
            train_type_3 = np.array(train_segments_set['interictal'])

            val_type_1 = np.array(val_segments_set['preictal_ontime'])
            val_type_3 = np.array(val_segments_set['interictal'])
            
            # scale_rate = 128
            # downsampling_factor = 2
            # patch_shape = (2,2)
            # full_model = one_channel_mobile_vit(
            #     image_size = (scale_rate, int(window_size * sr / downsampling_factor), 1),
            #     patch_shape = patch_shape
            # )
            inputs = Input(shape=(1,int(window_size*sr)))
            dilation_output = td_net(inputs)
            full_model = Model(inputs=inputs, outputs=dilation_output)

            checkpoint_path = f"./Dilation/{model_name}/{patient_name}/set{idx+1}/cp.ckpt"
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
                        with open(f'{checkpoint_dir}/ValResults', 'rb') as file_pi:
                            result_list = pickle.load(file_pi)
                            print(f'set{idx+1} Sensitivity : {result_list[2]}   FPR : {result_list[3]}')
                            patient_sens_sum += result_list[2]
                            patient_fpr_sum += result_list[3]
                        continue

            full_model.compile(optimizer = 'Adam',
                                metrics=[
                                        # tf.keras.metrics.BinaryAccuracy(threshold=0),
                                        # tf.keras.metrics.Recall(thresholds=0)
                                        # tf.keras.metrics.BinaryAccuracy(threshold=0),
                                        # tf.keras.metrics.Recall(thresholds=0)
                                        tf.keras.metrics.CategoricalAccuracy(),
                                        tf.keras.metrics.Recall(class_id=1),
                                        ] ,
                                #loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 
                                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
                                )

            if os.path.exists(checkpoint_path):
                print("Model Loaded!")
                full_model = tf.keras.models.load_model(checkpoint_path)

            logs = f"logs/{model_name}/{patient_name}/set{idx+1}"    

            tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                            histogram_freq = 1,
                                                            profile_batch = '100,200')
            
            # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', 
            #                                                     verbose=0,
            #                                                     patience=7,
            #                                                     mode='max',
            #                                                     )
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                                verbose=0,
                                                                patience=7,
                                                                )
            
            backup_callback = tf.keras.callbacks.BackupAndRestore(
            f"{checkpoint_dir}/training_backup",
            save_freq="epoch",
            delete_checkpoint=True,
            )
            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                            save_best_only=True,
                                                            monitor='val_categorical_accuracy',
                                                            verbose=0)
            
            # %%

            # train_generator = ViTGenerator_one_channel(type_1_data = train_type_1, 
            #                             type_3_data = train_type_3,
            #                             batch_size = batch_size,
            #                             data_type = data_type,
            #                             scale_resolution = scale_rate,
            #                             sampling_rate=sr,
            #                             ds_factor=downsampling_factor
            #                             )
            # validation_generator = ViTGenerator_one_channel(type_1_data = val_type_1,
            #                                     type_3_data = val_type_3, 
            #                                     batch_size = batch_size,
            #                                     data_type = data_type,
            #                                     scale_resolution = scale_rate,
            #                                     sampling_rate=sr,
            #                                     ds_factor=downsampling_factor
            #                                     )
            
            train_generator = FullModel_generator(type_1_data = train_type_1,
                                                  type_3_data = train_type_3,
                                                  batch_size = batch_size,
                                                  data_type = data_type)
            validation_generator = FullModel_generator( type_1_data = val_type_1,
                                                        type_3_data = val_type_3,
                                                        batch_size = batch_size,
                                                        data_type = data_type)

            history = full_model.fit(
                        train_generator,
                        epochs = epochs,
                        validation_data = validation_generator,
                        use_multiprocessing=True,
                        workers=28,
                        max_queue_size=80,
                        callbacks= [  cp_callback, early_stopping, backup_callback ]
            )

            del full_model
            #del dilation_output
            matrix, postprocessed_matrix, sens, fpr, seg_results = TestFullModel_specific.validation(checkpoint_path, val_intervals, data_type, 5,3, window_size=window_size)
            patient_sens_sum += sens
            patient_fpr_sum += fpr
            result_list = [matrix, postprocessed_matrix, sens, fpr, seg_results]
            print(f'{patient_name} set{idx+1},  Sensitivity : {sens}    FPR : {fpr}/h')
            print(f'{patient_name} Avg,         Sensitivity : {patient_sens_sum/(idx+1)} FPR : {patient_fpr_sum/(idx+1)}/h')
            
            
            
            

            with open(f'{checkpoint_dir}/training_done','w') as f:
                f.write('1')
            with open(f'{checkpoint_dir}/trainHistoryDict', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            with open(f'{checkpoint_dir}/ValResults', 'wb') as file_pi:
                pickle.dump(result_list, file_pi)

            
            SaveAsHeatmap(matrix, f"{checkpoint_dir}/categorical_matrix.png")
            plt.clf()
            SaveAsHeatmap(postprocessed_matrix, f"{checkpoint_dir}/tf_matrix.png" )   
            plt.clf()

           
        total_sens_sum += patient_sens_sum
        total_fpr_sum += patient_fpr_sum
        print(f'Total          Avg,         Sensitivity : {total_sens_sum/(patient_idx+1)} FPR : {total_fpr_sum/(patient_idx+1)}/h')
        patient_sens_sum = 0
        patient_fpr_sum = 0

def SaveAsHeatmap(matrix, path):
    sns.heatmap(matrix,annot=True, cmap='Blues')
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.savefig(path)
    plt.clf()




