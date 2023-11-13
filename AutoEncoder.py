from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, ZeroPadding2D, SeparableConv2D, UpSampling2D, BatchNormalization, ReLU, LeakyReLU
from tensorflow.keras.layers import AveragePooling1D, Flatten, Conv1DTranspose, Conv2DTranspose, Reshape, Concatenate, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Sequential

# Encoder

def FullChannelEncoder(inputs, window_size = 2, dilated = [1,2,4,8,16,32], freq = 256, 
					   filter_num=32, stride_num = 1, channel_num = 21, pooling_rate = 8):
	layers_list = []
	cnt = 0
	for df in dilated:
		x = Conv2D(filters=8, kernel_size = (1,4),padding='same',activation='relu',dilation_rate=(1,df))(inputs)	# (None, 21, 640, 8)
		x = MaxPooling2D((1,2))(x)	# (None, 21, 320, 8)
		x = BatchNormalization()(x)
		x = Conv2D(filters=8, kernel_size=(1,4),activation='relu',padding='same')(x)	# (None, 21, 320, 8)
		x = MaxPooling2D((1,2))(x)	# (None, 21, 160, 8)
		x = BatchNormalization()(x)
		x = Conv2D(filters=8, kernel_size=(1,4),activation='relu',padding='same')(x)	# (None, 21, 160, 8)
		x = MaxPooling2D((1,2))(x)	# (None, 21, 80, 8)
		x = BatchNormalization()(x)
		x = Conv2D(filters=8, kernel_size=(channel_num,1),activation='relu', padding='valid')(x)	# (None, 1, 80, 8)
		x = BatchNormalization()(x)
		layers_list.append(x)
		cnt+=1
	x = Concatenate(axis=-1)(layers_list) # (None, 1, 80, 48)
	x=  tf.squeeze(x, axis = -3) # (None, 80, 48)
	return x


def FullChannelDecoder(inputs, dilated = [1,2,4,8,16,32], pooling_rate = 8, freq = 256, window_size = 2):
	x_input = Reshape((1,inputs.shape[1],inputs.shape[2]))(inputs)
	x_list = []
	cnt = 0
	sum = 0
	for i in range(len(dilated)):
		x = Conv2DTranspose(filters=8,kernel_size=(21,1),activation='relu',padding='valid')(x_input[:,:,:,8*i:8*(i+1)])
		x = BatchNormalization()(x)
		x = UpSampling2D(size=(1,2))(x)
		x = Conv2DTranspose(filters=8,kernel_size=(1,4),activation='relu',padding='same')(x)
		x = BatchNormalization()(x)
		x = UpSampling2D(size=(1,2))(x)
		x = Conv2DTranspose(filters=8,kernel_size=(1,4),activation='relu',padding='same')(x)
		x = BatchNormalization()(x)
		x = UpSampling2D(size=(1,2))(x)
		x = Conv2DTranspose(filters=8,kernel_size=(1,4),padding='same',activation='relu',dilation_rate=dilated[i])(x)
		x_list.append(x)
		cnt+=1
		
	x = Concatenate(axis=-1)(x_list)
	x = x[:,:,-1*freq*window_size:,:]
	decoder_output = Conv2DTranspose(filters=1, kernel_size=(1,4),padding='same')(x)

	return decoder_output

def FullChannelEncoder_paper_base(inputs):

	#inputs = (None, 21, 640, 1)
	x = Conv2D(filters=8, kernel_size = (2,1),padding='valid')(inputs)	# (None, 20, 640, 32)
	x = LeakyReLU()(x)
	#x = BatchNormalization()(x)
	x = MaxPooling2D((2,2))(x)	# (None, 10, 320, 32)
	
	x = Conv2D(filters=16, kernel_size=(2,3),padding='same')(x)	# (None, 10, 320, 32)
	x = LeakyReLU()(x)
	x = MaxPooling2D((2,2))(x)	# (None, 5, 160, 32)
	
	x = Conv2D(filters=32, kernel_size=(2,3),padding='valid')(x)	# (None, 4, 158, 32)
	x = LeakyReLU()(x)
	x = MaxPooling2D((2,2))(x)	# (None, 2, 79, 32)
	
	x = Conv2D(filters=64, kernel_size=(2,3), padding='valid')(x)	# (None, 1, 77, 32)
	x=  tf.squeeze(x, axis = -3, name="encoder_last") # (None, 77, 32)


	return x

def FullChannelDecoder_paper_base(inputs):
	# (None, 77, 32)
	x_input = Reshape((1,inputs.shape[1],inputs.shape[2]))(inputs)
	
	x = Conv2DTranspose(filters=32,kernel_size=(2,3), padding='valid')(x_input)# (None, 2, 79, 32)
	x = UpSampling2D(size=(2,2))(x)	
	x = LeakyReLU()(x)															# (None, 4, 158, 32)
	

	x = Conv2DTranspose(filters=16,kernel_size=(2,3),padding='valid')(x)		# (None, 5, 160, 32)
	x = UpSampling2D(size=(2,2))(x)																# (None, 10, 320, 32)
	x = LeakyReLU()(x)

	x = Conv2DTranspose(filters=8,kernel_size=(2,3),padding='same')(x)		# (None, 10, 320, 32)
	x = UpSampling2D(size=(2,2))(x)																# (None, 20, 640, 32
	#x = BatchNormalization()(x)
	x = LeakyReLU()(x)
	
	x = Conv2DTranspose(filters=1,kernel_size=(2,1),padding='valid')(x) # (None, 21, 640, 1)
	decoder_output = tf.squeeze(x, axis = -1)

	return decoder_output





# inputs = Input(shape=(21,512,1))
# encoder_output = FullChannelEncoder(64,inputs)

# decoder_output = FullChannelDecoder(encoder_output)
# model = Model(inputs=inputs, outputs=decoder_output)
# model.summary()


#tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)




