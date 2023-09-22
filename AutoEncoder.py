from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import Sequential

# Encoder

def FullChannelEncoder(encoded_feature_num,inputs, window_size = 2, dilated = [1,2,4,8,16,32], freq = 256, 
					   filter_num=32, stride_num = 1, channel_num = 23, pooling_rate = 8):
	layers_list = []
	for df in dilated:
		x = SeparableConv2D(filters=16, kernel_size = (3,16), activation='relu',padding='same',dilation_rate=(1,df))(inputs)
		padded = ZeroPadding2D(padding=( (0,0),(df*(8-1),0) ))(x)
		layers_list.append(SeparableConv2D(filters=2, kernel_size=(23,8), activation='relu', padding='valid',dilation_rate=(1,df))(padded))
	x = Concatenate(axis=-1)(layers_list)

	x = MaxPooling2D((1,pooling_rate))(x)
	x = Reshape((int(window_size*freq/pooling_rate),len(dilated)*2))(x)

	return x

def FullChannelDecoder(inputs, dilated = [1,2,4,8,16,32], pooling_rate = 8):
	x = Reshape((1,inputs.shape[1],inputs.shape[2]))(inputs)
	x_upsampled = UpSampling2D(size=(1,pooling_rate))(x)
	x_splited = tf.split(x_upsampled,len(dilated),axis=-1)
	x_list = []
	print(x_splited)
	for i in range(len(dilated)):
		x_list.append(  Conv2DTranspose(filters=16,kernel_size=(23,8),activation='relu')(x_splited[i]) )
	print(x_list)

	

inputs = Input(shape=(23,512,1))
encoder_output = FullChannelEncoder(64,inputs)

model = Model(inputs = inputs, outputs = encoder_output)
model.summary()
inputs = Input(shape=(64,12))
FullChannelDecoder(inputs)



"""
inputs = Input(shape=(23,1280,1))
df = 8

outputs = Conv2D(filters=2,kernel_size=(23,8),padding='valid',dilation_rate=(1,df))(inputs_pad)
outputs = Reshape((1280,2,1))(outputs)
outputs = AveragePooling2D((16,1))(outputs)

models = Model(inputs=inputs,outputs=outputs)
models.summary()
"""





