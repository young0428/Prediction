from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D, BatchNormalization, ReLU, LeakyReLU, Bidirectional
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D, LSTM
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential


# Generator : InterIctal rawEEG 들어가면 Preictal CWT Image가 출력되는 모델 하나, Preictla rawEEG가 들어가면 Interictal rawEEG가 출력되는 모델 하나를 만듦
# Sampling rate = 200, window_duration = 5 기준, (1000, 1) -> (1000, 64) -> (64, 1000)

def Build_Generator(inputs):
    sampling_rate = 200
    window_duration = 5
    window_size = sampling_rate * window_duration
    
    x = inputs
    
    x = Conv1D(64, kernel_size = 8, dilation_rate = 1, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv1D(64, kernel_size = 8, dilation_rate = 2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv1D(64, kernel_size = 8, dilation_rate = 4, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv1D(64, kernel_size = 8, dilation_rate = 8, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    
    x = tf.transpose(x,[0,2,1])
    
    return x


# 30초 짜리 CWT image를 가지고 판단
def Build_Discriminator(inputs):
    
    # (64, 1000)
    x = tf.expand_dims(inputs, axis=-1)
    
    x = Conv2D(16, kernel_size = (2,4), padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = LeakyReLU(0.2)(x)
    
    # (32, 500, 16)
    x = Conv2D(32, kernel_size = (2,4), padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = LeakyReLU(0.2)(x)
    
    # (16, 250, 32)
    x = Conv2D(32, kernel_size = (2,4), padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = LeakyReLU(0.2)(x)
    
    # (8, 125, 32)
    x = Conv2D(32, kernel_size = (3,6), padding='valid')(x)
    # (6, 120, 32)
    x = MaxPooling2D((2,2))(x)
    x = LeakyReLU(0.2)(x)
    
    # (3, 60, 32)
    x = Conv2D(32, kernel_size = (3,6), padding='valid')(x)
    x = tf.squeeze(x, axis = -3)
    # (55, 32)
    
    x = Bidirectional(LSTM(64))(x)
    # (40)
    disc_output = Dense(64, activation=tf.nn.gelu)(x)
    disc_output = Dense(1,activation='sigmoid')(disc_output)
    
    classifier_output = Dense(64, activation=tf.nn.gelu)(x)
    classifier_output = Dense(1, activation='sigmoid')(classifier_output)
    
    return disc_output, classifier_output


gen_inputs = Input(shape=(1000,1))
generator_i2p = Build_Generator(gen_inputs)
i2p_model = Model(gen_inputs, generator_i2p)
generator_p2i = Build_Generator(gen_inputs)
p2i_model = Model(gen_inputs, generator_p2i)

disc_inputs = Input(shape=(64,1000))
disc_output, classifier_output = Build_Discriminator(disc_inputs)
disc_model = Model(disc_inputs, [disc_output, classifier_output])


disc_model.summary()




    
    

    
    
    
    