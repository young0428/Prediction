from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import Sequential

# Encoder

def FullChannelEncoder(encoded_feature_num,inputs, window_size = 2, dilated = [1,2,4,8,16,32], freq = 256, 
					   filter_num=32, stride_num = 1, channel_num = 21, pooling_rate = 8):
	layers_list = []
	for df in dilated:
		x = Conv2D(filters=8, kernel_size = (1,4), activation='relu',padding='same',dilation_rate=(1,df))(inputs)	# (None, 23, 512, 8)
		#padded = ZeroPadding2D(padding=( (0,0),(df*(8-1),0) ))(x)
		x = MaxPooling2D((1,2))(x)	# (None, 23, 256, 8)
		x = Conv2D(filters=16, kernel_size=(1,4),activation='relu',padding='same')(x)	# (None, 23, 256, 16)
		x = MaxPooling2D((1,4))(x)	# (None, 23, 128, 16)
		#x = Conv2D(filters=16, kernel_size=(1,4),activation='relu',padding='same')(x)	# (None, 23, 128, 32)
		#x = MaxPooling2D((1,2))(x)	# (None, 23, 64, 32)
		x = Conv2D(filters=2, kernel_size=(channel_num,1), activation='relu', padding='valid')(x)	# (None, 1, 64, 2)
		layers_list.append(x)
	x = Concatenate(axis=-1)(layers_list)
	x=  tf.squeeze(x, axis = -3) # (None, 64, 12)
	x = Flatten()(x)
	x = Dense(128)(x)
	#x = Reshape((int(window_size*freq/pooling_rate),len(dilated)*2))(x)

	return x



def FullChannelDecoder(inputs, dilated = [1,2,4,8,16,32], pooling_rate = 8, freq = 256, window_size = 2):
	x = Dense(64*12)(inputs)
	x = Reshape((1,64,12))(x)
	#x = Reshape((1,inputs.shape[1],inputs.shape[2]))(inputs)
	x_splited = tf.split(x,len(dilated),axis=-1)
	x_list = []
	for i in range(len(dilated)):
		x = Conv2DTranspose(filters=2,kernel_size=(21,1),activation='relu',padding='valid')(x_splited[i])
		#x = UpSampling2D(size=(1,2))(x)
		#x = Conv2DTranspose(filters=16,kernel_size=(1,4),activation='relu',padding='same')(x)
		x = UpSampling2D(size=(1,4))(x)
		x = Conv2DTranspose(filters=16,kernel_size=(1,4),activation='relu',padding='same')(x)
		x = UpSampling2D(size=(1,2))(x)
		x = Conv2DTranspose(filters=8,kernel_size=(1,4),activation='relu',padding='same',dilation_rate=dilated[i])(x)
		x_list.append(x)

	x = Concatenate(axis=-1)(x_list)
	x = x[:,:,-1*freq*window_size:,:]
	decoder_output = Conv2DTranspose(filters=1, kernel_size=(1,2),activation='relu',padding='same')(x)

	return decoder_output



# inputs = Input(shape=(21,512,1))
# encoder_output = FullChannelEncoder(64,inputs)

#decoder_output = FullChannelDecoder(encoder_output)
#model = Model(inputs=inputs, outputs=decoder_output)
#model.summary()


#tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)




