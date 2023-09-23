from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D, TimeDistributed
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import Sequential
from AutoEncoder import FullChannelDecoder, FullChannelEncoder
from LSTMmodel import LSTMLayer


encoder_input = Input(shape = (23,512,1))
encoder_output = FullChannelEncoder(64,encoder_input)
encoder_model = Model(inputs=encoder_input, outputs=encoder_output)
ts_input = Input(shape=(10,23,512,1))
ts_encoder_output = TimeDistributed(encoder_model)(ts_input)
prediction = LSTMLayer(ts_encoder_output)

model = Model(inputs=ts_input,outputs=prediction)
model.summary()

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
