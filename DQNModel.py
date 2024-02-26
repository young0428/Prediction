from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D, BatchNormalization, ReLU, LeakyReLU, Bidirectional
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D, LSTM
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from ViTModel import BoradAttentionViT
from tensorflow.keras import Sequential


# Generator : InterIctal rawEEG 들어가면 Preictal CWT Image가 출력되는 모델 하나, Preictla rawEEG가 들어가면 Interictal rawEEG가 출력되는 모델 하나를 만듦
# Sampling rate = 200, window_duration = 5 기준, (1000, 1) -> (1000, 64) -> (64, 1000)
class NoisyDense(tf.keras.layers.Layer):
    """ Factorized Gaussian Noisy Dense Layer"""
    def __init__(self, units, activation=None, trainable=True):
        super(NoisyDense, self).__init__()
        self.units = units
        self.trainable = trainable
        self.activation = tf.keras.activations.get(activation)
        self.sigma_0 = 0.5

    def build(self, input_shape):
        p = input_shape[-1]
        self.w_mu = self.add_weight(
            name="w_mu", shape=(int(input_shape[-1]), self.units),
            initializer=tf.keras.initializers.RandomUniform(-1. / np.sqrt(p), 1. / np.sqrt(p)),
            trainable=self.trainable)

        self.w_sigma = self.add_weight(
            name="w_sigma", shape=(int(input_shape[-1]), self.units),
            initializer=tf.keras.initializers.Constant(self.sigma_0 / np.sqrt(p)),
            trainable=self.trainable)

        self.b_mu = self.add_weight(
            name="b_mu", shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(-1. / np.sqrt(p), 1. / np.sqrt(p)),
            trainable=self.trainable)

        self.b_sigma = self.add_weight(
            name="b_sigma", shape=(self.units,),
            initializer=tf.keras.initializers.Constant(self.sigma_0 / np.sqrt(p)),
            trainable=self.trainable)

    def call(self, inputs, noise=True):

        epsilon_in = self.f(tf.random.normal(shape=(self.w_mu.shape[0], 1), dtype=tf.float32))
        epsilon_out = self.f(tf.random.normal(shape=(1, self.w_mu.shape[1]), dtype=tf.float32))

        w_epsilon = tf.matmul(epsilon_in, epsilon_out)
        b_epsilon = epsilon_out

        w = self.w_mu + self.w_sigma * w_epsilon
        b = self.b_mu + self.b_sigma * b_epsilon

        out = tf.matmul(inputs, w) + b
        if self.activation is not None:
            out = self.activation(out)
        return out

    @staticmethod
    def f(x):
        x = tf.sign(x) * tf.sqrt(tf.abs(x))
        return x
def Build_DQN_model(cwt_inputs, action_state_inputs , num_atom):
    
    num_action = 2
    Vmin = -10
    Vmax = 10
    z = np.linspace(Vmin, Vmax, num_atom, dtype=np.float32)
    
    x = cwt_inputs

    cwt_feature = BoradAttentionViT(x)

    action_state_feature = tf.tile(action_state_inputs, [1, 10])
    total_feature = tf.concat([action_state_feature, cwt_feature],axis=-1)
    
    value_stream  = NoisyDense(num_atom, activation='relu')(total_feature)
    
    advantage_stream = []
    for i in range(num_action):
        advantage_stream.append(NoisyDense(num_atom, activation='relu')(total_feature))
    
    value = tf.reshape(value_stream, [-1, 1, num_atom])
    advantage = tf.reshape(advantage_stream, [-1, num_action, num_atom])
    
    
    q_atoms = value + (advantage - tf.reduce_mean(advantage, axis = 1, keepdims=True, name='min_of_action_value'))
    distribution = tf.nn.softmax(q_atoms, axis = -1)
    return Model([cwt_inputs, action_state_inputs], distribution)


if __name__ == '__main__':
    import time
    cwt_inputs = tf.keras.layers.Input(shape=(128,4000, 1))
    action_state_inputs = tf.keras.layers.Input(shape=(1))
    model = Build_DQN_model(cwt_inputs, action_state_inputs, 51)
    model.summary()
    time.sleep(5)
    
    








    
    

    
    
    
    