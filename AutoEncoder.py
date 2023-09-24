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
		x = SeparableConv2D(filters=8, kernel_size = (1,8), activation='relu',padding='same',dilation_rate=(1,df))(inputs)
		#padded = ZeroPadding2D(padding=( (0,0),(df*(8-1),0) ))(x)
		x = SeparableConv2D(filters=2, kernel_size=(23,1), activation='relu', padding='valid')(x)
		x = MaxPooling2D((1,2))(x)
		x = SeparableConv2D(filters=2, kernel_size=(1,4),activation='relu',padding='same')(x)
		x = MaxPooling2D((1,4))(x)
		layers_list.append(x)
	x = Concatenate(axis=-1)(layers_list)
	x = Reshape((int(window_size*freq/pooling_rate),len(dilated)*2))(x)

	return x

def FullChannelDecoder(inputs, dilated = [1,2,4,8,16,32], pooling_rate = 8):
	x = Reshape((1,inputs.shape[1],inputs.shape[2]))(inputs)
	
	x_splited = tf.split(x,len(dilated),axis=-1)
	x_list = []
	for i in range(len(dilated)):
		x = UpSampling2D(size=(1,4))(x_splited[i])
		x = Conv2DTranspose(filters=2,kernel_size=(1,4),activation='relu',padding='same')(x)
		x = UpSampling2D(size=(1,2))(x)
		x = Conv2DTranspose(filters=2,kernel_size=(23,1),activation='relu',padding='valid')(x)
		x = Conv2DTranspose(filters=8,kernel_size=(1,16),activation='relu',padding='same',dilation_rate=dilated[i])(x)
		x_list.append(x)

	x = Concatenate(axis=-1)(x_list)
	x = x[:,:,-1*256*2:,:]
	decoder_output = Conv2DTranspose(filters=1, kernel_size=(1,2),activation='relu',padding='same')(x)

	return decoder_output



"""
inputs = Input(shape=(23,512,1))
encoder_output = FullChannelEncoder(64,inputs)
decoder_output = FullChannelDecoder(encoder_output)
model = Model(inputs=inputs, outputs=decoder_output)
model.summary()
"""

#tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)




