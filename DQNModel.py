from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D, BatchNormalization, ReLU, LeakyReLU, Bidirectional
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D, LSTM
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from ViTModel import BoradAttentionViT
from tensorflow.keras import Sequential


# Generator : InterIctal rawEEG 들어가면 Preictal CWT Image가 출력되는 모델 하나, Preictla rawEEG가 들어가면 Interictal rawEEG가 출력되는 모델 하나를 만듦
# Sampling rate = 200, window_duration = 5 기준, (1000, 1) -> (1000, 64) -> (64, 1000)

def Build_DQN_model(cwt_inputs, action_state_inputs , num_atom):
    
    num_action = 2
    Vmin = -10
    Vmax = 10
    z = np.linspace(Vmin, Vmax, num_atom, dtype=np.float32)
    
    x = cwt_inputs
    cwt_feature = BoradAttentionViT(x)
    action_state_feature = Dense(10)(action_state_inputs)
    total_feature = tf.concat([action_state_feature, cwt_feature],axis=-1)
    
    value_stream  = Dense(num_atom,name='value_stream')(total_feature)
    
    advantage_stream = []
    for i in range(num_action):
        advantage_stream.append(Dense(num_atom, activation='softmax', name=f"action_{i}_distribution")(total_feature))
        
    value = tf.reshape(value_stream, [-1, 1, num_atom], name='reshaped_value')
    advantage = tf.reshape(advantage_stream, [-1, num_action, num_atom], name='reshaped_advantage')
    

    value = tf.reduce_sum(tf.multiply(z, value),axis=-1, name='value_expectation')
    advantage = tf.reduce_sum(tf.multiply(z, advantage), axis=-1, name='advantage_expectation')
    
    
    q_values = value + (advantage - tf.reduce_mean(advantage, axis = -1, keepdims=True, name='min_of_action_value'))
    
    return Model([cwt_inputs, action_state_inputs], q_values)

if __name__ == '__main__':
    cwt_inputs = tf.keras.layers.Input(shape=(128,6000, 1))
    action_state_inputs = tf.keras.layers.Input(shape=(1))
    model = Build_DQN_model(cwt_inputs, action_state_inputs, 51)
    model.summary()








    
    

    
    
    
    