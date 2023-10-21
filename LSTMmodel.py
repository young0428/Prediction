from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, LSTM, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D, TimeDistributed
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential

def LSTMLayer(inputs,cell_num = 16):
    # Flatten

    x = Reshape((10,64*21))(inputs)
    #x = Reshape((10*64*21))(inputs)
    x = Dense(32)(x)
    x = Dropout(0.2)(x)
    
    x = LSTM(cell_num)(x)
    x = Dense(1,activation='sigmoid')(x)

    return x


    

# autoencoder_model_path = "AutoEncoder_training_0/cp.ckpt"

# encoder_inputs = Input(shape=(21,512,1))
# encoder_outputs = FullChannelEncoder(encoded_feature_num=64,inputs = encoder_inputs)
# decoder_outputs = FullChannelDecoder(encoder_outputs)
# autoencoder_model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
# autoencoder_model.compile(optimizer = 'Adam', loss='mse',)
# autoencoder_model.load_weights(autoencoder_model_path)
# autoencoder_model.trainable = False
# encoder_model = Model(inputs=encoder_inputs, outputs=encoder_outputs)








    
