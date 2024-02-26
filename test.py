import tensorflow as tf
import numpy as np
#from DQNModel import Build_DQN_model
import time


cwt_inputs = tf.keras.layers.Input(shape=(20))
a = tf.keras.layers.Dense(1)(cwt_inputs)
model = tf.keras.Model(inputs=cwt_inputs,outputs = a)
model.summary()
time.sleep(5)