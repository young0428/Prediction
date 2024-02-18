from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D, BatchNormalization, ReLU, LeakyReLU, Bidirectional
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D, LSTM
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential


# (None, 128, 12000) -> (None, 30 , 128, 400)
# input으로 CWT 이미지를 일정 시간 간격마다 자른 것을 입력으로 받음
# (128, 400,1)
def PreCNN_LSTM(inputs):
    
    #x = Reshape((inputs.shape[1],inputs.shape[2]))(inputs)
    x = Conv2D(filters=32,kernel_size=(2,5), padding='same', activation=tf.nn.gelu)(inputs)
    x = MaxPooling2D((2,2))(x)
    # (64, 200, 32)
    
    x = Conv2D(filters=64,kernel_size=(2,5), padding='same', activation=tf.nn.gelu)(x)
    x = MaxPooling2D((2,2))(x)
    # (32, 100, 64)
    
    x = Conv2D(filters=128,kernel_size=(2,5), padding='same', activation=tf.nn.gelu)(x)
    x = MaxPooling2D((2,2))(x)
    # (16, 50, 128)
    
    x = Conv2D(filters=1,kernel_size=(2,5), padding='same', activation=tf.nn.gelu)(x)
    # (16, 50, 1)
    x = tf.squeeze(x, axis = -1)
    # (16, 50)
    x = tf.transpose(x,[0,2,1])
    # (50, 16)
    x = Bidirectional(LSTM(32))(x)
    # (64)

    return x

def PostLSTM(inputs, sr = 200, splited_window_size = 2, scale_resolution = 128):
    
    cl_inputs = Input(shape=(scale_resolution, sr*splited_window_size,1))
    cl_outputs = PreCNN_LSTM(cl_inputs)
    pre_cl_model = Model(inputs=cl_inputs, outputs=cl_outputs)
    
    
    # (None, 128, 12000) -> (None, 128, 30, 400, 1)
    x = Reshape((inputs.shape[1], int(inputs.shape[2]/(sr*splited_window_size)), sr*splited_window_size, 1))(inputs)
    x = tf.transpose(x, [0, 2, 1, 3, 4])
    x = tf.keras.layers.TimeDistributed(pre_cl_model)(x)
    x = Dropout(0.1)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
    x = Dropout(0.1)(x)
    x = Dense(2, activation='softmax')(x)
    
    return x






    
    
    
    