# %%
import os
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.layers import Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import F1Score, Recall, Precision

from ModelGenerator import *
from readDataset import *
from LSTMmodel import LSTMLayer
from sklearn.model_selection import KFold
from vit_tensorflow.mobile_vit import one_channel_mobile_vit
from ViTModel import BuildMViTModel


tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

# %%
def train(model_name, data_type = 'snu'):
    epochs = 100
    batch_size = 48 # 한번의 gradient update시마다 들어가는 데이터의 사이즈
    window_size = 8
    overlap_sliding_size = 2
    normal_sliding_size = window_size
    sampling_rate = 200
    scale_rate = 128
    patch_shape = (2, 2)
    downsampling_factor = 2

    checkpoint_path = f"ViT/{model_name}/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']
    # %%
    if data_type=='snu':
        train_info_file_path = "/host/d/SNU_DATA/patient_info_snu_train.csv"
        test_info_file_path = "/host/d/SNU_DATA/patient_info_snu_test.csv"
        edf_file_path = "/host/d/SNU_DATA"

        if os.path.exists(checkpoint_dir):
            print("Model Loaded")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            try:
                os.makedirs(checkpoint_dir)
            except:
                pass
            

    elif data_type=='chb':
        train_info_file_path = "/host/d/CHB/patient_info_chb_train.csv"
        test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
        edf_file_path = "/host/d/CHB"
        
        if os.path.exists(checkpoint_path):
            print("#####################")
            print("### Model Loaded! ###")
            print("#####################")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            
            try:
                os.makedirs(checkpoint_dir)
            except:
                pass
            inputs = tf.keras.layers.Input(shape=(18, scale_rate, sampling_rate*window_size, 1))
            
            model = BuildMViTModel(inputs)

    elif data_type=='snu_one_ch':

        train_info_file_path = "/host/d/SNU_DATA/patient_info_snu_train.csv"
        test_info_file_path = "/host/d/SNU_DATA/patient_info_snu_test.csv"
        edf_file_path = "/host/d/SNU_DATA"
        
        if os.path.exists(checkpoint_path):
            print("#####################")
            print("### Model Loaded! ###")
            print("#####################")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            try:
                os.makedirs(checkpoint_dir)
            except:
                pass
            model = one_channel_mobile_vit(
                image_size = (scale_rate, int(window_size * sampling_rate / downsampling_factor), 1),
                patch_shape = patch_shape
            )
    elif data_type=='chb_one_ch':
        train_info_file_path = "/host/d/CHB/patient_info_chb_train.csv"
        test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
        edf_file_path = "/host/d/CHB"
        
        if os.path.exists(checkpoint_path):
            print("#####################")
            print("### Model Loaded! ###")
            print("#####################")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            try:
                os.makedirs(checkpoint_dir)
            except:
                pass
            model = one_channel_mobile_vit(
                image_size = (scale_rate, int(window_size * sampling_rate / downsampling_factor), 1),
                patch_shape = patch_shape
            )


    model.summary()
    _ , train_interval_overall = LoadDataset(train_info_file_path)
    train_segments_set = {}
    test_segments_set = {}
    # %%
    # train interval 가져옴
    # 해당하는 채널이 전부 있는 interval만 필터링
    channel_filtered_intervals = FilteringByChannel(train_interval_overall, edf_file_path, data_type)
    # 환자 이름에 따라 key 만들어서 dictionary 형태로 interval 보관
    interval_dict_key_patient_name = Interval2NameKeyDict(channel_filtered_intervals)
    # ictal이 2번 이상인 환자만 남겨놓음
    filtered_interval_dict, ictal_num = FilterValidatePatient(interval_dict_key_patient_name)

    train_interval_set = []
    for patient_name in filtered_interval_dict.keys():
        train_interval_set += filtered_interval_dict[patient_name]

    # %%
    # Test Interval 가져옮
    _ , test_interval_overall = LoadDataset(test_info_file_path)
    test_interval_set = FilteringByChannel(test_interval_overall, edf_file_path, data_type)

    #%%
    train_interval_set = IntervalList2Dict(train_interval_set)
    test_interval_set = IntervalList2Dict(test_interval_set)

    #%%
    # 상대적으로 데이터 갯수가 적은 것들은 window_size 2초에 sliding_size 1초로 overlap 시켜 데이터 증강
    for state in ['preictal_ontime', 'preictal_late']:
        train_segments_set[state] = Interval2Segments(train_interval_set[state],edf_file_path, window_size, overlap_sliding_size)
        test_segments_set[state] = Interval2Segments(test_interval_set[state],edf_file_path, window_size, overlap_sliding_size)
        
    for state in ['interictal']:
        train_segments_set[state] = Interval2Segments(train_interval_set[state],edf_file_path, window_size, normal_sliding_size)
        test_segments_set[state] = Interval2Segments(test_interval_set[state],edf_file_path, window_size, normal_sliding_size)

    train_type_1 = np.array(train_segments_set['preictal_ontime'] + train_segments_set['preictal_late'])
    #train_type_2 = np.array(train_segments_set['ictal'])
    train_type_3 = np.array(train_segments_set['interictal'])

    test_type_1 = np.array(test_segments_set['preictal_ontime'] + test_segments_set['preictal_late'] )
    #test_type_2 = np.array(test_segments_set['ictal'])
    test_type_3 = np.array(test_segments_set['interictal'])



    # model.compile(optimizer = 'Adam',
    #                     metrics=[
    #                             tf.keras.metrics.BinaryAccuracy(threshold=0),
    #                             tf.keras.metrics.Recall(thresholds=0)
    #                             ] ,
    #                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.05) )
    model.compile(optimizer = 'Adam',
                        metrics=[
                                tf.keras.metrics.CategoricalAccuracy(),
                                tf.keras.metrics.Recall(class_id=1)
                                ] ,
                        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05))

    logs = f"/ViT/{model_name}/logs/"    

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                    histogram_freq = 1,
                                                    profile_batch = '100,200')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', 
                                                        verbose=0,
                                                        patience=10,
                                                        restore_best_weights=True)
    backup_callback = tf.keras.callbacks.BackupAndRestore(
        f"./ViT/{model_name}/training_backup",
        save_freq="epoch",
        delete_checkpoint=True,
    )

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_best_only=True,
                                                    verbose=0)
    train_generator = ViTGenerator_one_channel(type_1_data = train_type_1, 
                                        type_3_data = train_type_3,
                                        batch_size = batch_size,
                                        data_type = data_type,
                                        scale_resolution = scale_rate,
                                        sampling_rate=sampling_rate,
                                        ds_factor=downsampling_factor
                                        )
    test_generator = ViTGenerator_one_channel(type_1_data = test_type_1,
                                        type_3_data = test_type_3, 
                                        batch_size = batch_size,
                                        data_type = data_type,
                                        scale_resolution = scale_rate,
                                        sampling_rate=sampling_rate,
                                        ds_factor=downsampling_factor
                                        )

    
    history = model.fit(
                train_generator,
                epochs = epochs,
                validation_data = test_generator,
                use_multiprocessing=True,
                workers=28,
                max_queue_size=30,
                shuffle=True,
                callbacks= [ tboard_callback, cp_callback, early_stopping, backup_callback ]
                )

    with open(f'./ViT/{model_name}/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


#%%
#train(lstm_model_name,encoder_model_name,'chb')
