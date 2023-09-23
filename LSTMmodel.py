from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, LSTM, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import Sequential

def LSTMLayer(inputs,cell_num = 16):
    # Flatten
    x = Reshape((10,64*12))(inputs)
    x = Dense(32)(x)

    
    x = LSTM(cell_num)(x)
    x = Dense(1,activation='sigmoid')(x)

    return x
    
