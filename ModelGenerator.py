import numpy as np
import tensorflow as tf
from keras.utils import Sequence
from readDataset import *
import PreProcessing
import random



class FullModel_generator(Sequence):
    def __init__(self,type_1_data=[], type_2_data=[], type_3_data=[], batch_size=500, data_type = 'snu', target_batch_num = None, gen_type = 'train'):
        
        self.ratio_type_1 = [5,4,3,2]
        self.ratio_type_2 = [0.05,0.05,0.05,0.05]
        self.ratio_type_3 = [5,6,7,8]
        self.batch_size = batch_size
        self.epoch = 0
        self.update_period = 10
        self.type_1_data = np.array(type_1_data)
        self.type_2_data = np.array(type_2_data)
        self.type_3_data = np.array(type_3_data)
        self.type_1_len = len(type_1_data)
        self.type_2_len = len(type_2_data)
        self.type_3_len = len(type_3_data)
        self.data_type = data_type
        self.gen_type = gen_type

        self.iden_mat = np.eye(2)
        
        
        self.batch_set, self.batch_num = updateDataSet(self.type_1_len, self.type_2_len, self.type_3_len, [self.ratio_type_1[0], self.ratio_type_2[0], self.ratio_type_3[0]], self.batch_size)
        if target_batch_num == None or gen_type == 'val':
            self.shuffled_batch_num = self.batch_num
        else:
            self.shuffled_batch_num = target_batch_num
        
        self.batch_mask = list(range(self.batch_num))
        self.batch_mask += [random.randint(0,self.batch_num-1) for _ in range(self.shuffled_batch_num - self.batch_num)]
        random.shuffle(self.batch_mask)
    def on_epoch_end(self):
        self.epoch += 1
        if self.epoch/self.update_period < 4:
            self.ratio_idx = int(self.epoch/self.update_period)
        else:
            self.ratio_idx = 3
        self.ratio_idx = 0
        self.batch_set, self.batch_num = updateDataSet(self.type_1_len, self.type_2_len, self.type_3_len, [self.ratio_type_1[self.ratio_idx], self.ratio_type_2[self.ratio_idx], self.ratio_type_3[self.ratio_idx]], self.batch_size)        
    
    def __len__(self):
        return self.shuffled_batch_num
    
    def __getitem__(self, idx):
        batch_concat = None
        origin_idx = self.batch_mask[idx]
        
        if len(self.batch_set[1][origin_idx]) == 0:
            type_1_seg = []
        else:
            type_1_seg = self.type_1_data[self.batch_set[0][self.batch_set[1][origin_idx]]]
            batch_concat = type_1_seg
  
        if len(self.batch_set[3][origin_idx]) == 0:
            type_2_seg = []
        else:
            type_2_seg = self.type_2_data[self.batch_set[2][self.batch_set[3][origin_idx]]]
            batch_concat = np.concatenate((batch_concat, type_2_seg))
        if len(self.batch_set[5][origin_idx]) == 0:
            type_3_seg = []
        else:
            type_3_seg = self.type_3_data[self.batch_set[4][self.batch_set[5][origin_idx]]]
            batch_concat = np.concatenate((batch_concat, type_3_seg))
            
        x_batch = Segments2Data(batch_concat, self.data_type)
        #x_batch = PreProcessing.FilteringSegments(x_batch)

        
        type_1_len = len(type_1_seg)
        type_2_len = len(type_2_seg)
        type_3_len = len(type_3_seg)
       
        y_batch = np.concatenate(( np.ones(type_1_len), 
                                   np.zeros(type_2_len), 
                                   np.zeros(type_3_len)))
        
        y_batch = np.asarray(y_batch, dtype=np.int32)
        y_batch = self.iden_mat[y_batch]
        

        return x_batch, y_batch
    
class SplitedLSTM_generator(FullModel_generator):
    def __init__(self,type_1_data=[], type_2_data=[], type_3_data=[], batch_size=500, data_type = 'snu', target_batch_num = None, gen_type = 'train', splited_window_size=2, channel=None):
        super().__init__(type_1_data, type_2_data, type_3_data, batch_size, data_type, target_batch_num, gen_type)
        self.splited_window_size = splited_window_size
        self.channel = channel
    def __getitem__(self, idx):
        batch_concat = None
        origin_idx = self.batch_mask[idx]

        if len(self.batch_set[1][origin_idx]) == 0:
            type_1_seg = []
        else:
            type_1_seg = self.type_1_data[self.batch_set[0][self.batch_set[1][origin_idx]]]
            batch_concat = type_1_seg

        if len(self.batch_set[3][origin_idx]) == 0:
            type_2_seg = []
        else:
            type_2_seg = self.type_2_data[self.batch_set[2][self.batch_set[3][origin_idx]]]
            batch_concat = np.concatenate((batch_concat, type_2_seg))
        if len(self.batch_set[5][origin_idx]) == 0:
            type_3_seg = []
        else:
            type_3_seg = self.type_3_data[self.batch_set[4][self.batch_set[5][origin_idx]]]
            batch_concat = np.concatenate((batch_concat, type_3_seg))

     
        x_batch = Segments2Data(batch_concat, self.data_type, manual_channels=self.channel)
        
        sr = 200
        # (batch_size, ch, window_size * sr)
        x_shape = x_batch.shape
        window_num = int(x_shape[2]/(sr*self.splited_window_size))
        x_batch.shape = (x_shape[0], x_shape[1], window_num, sr*self.splited_window_size)
        #(batch, ch, window_num, window_size * sr)
        if self.gen_type == 'train' and self.epoch <= 20:
            
            for i in range(x_shape[0]):
                one_data = x_batch[i]
                #(ch, window_num, window_size * sr)
                transposed = np.transpose(one_data,(1,0,2))
                #(window_num, ch, window_size * sr)
                np.random.shuffle(transposed)
                transposed = np.transpose(transposed,(1,0,2))
                x_batch[i] = transposed
        
        x_batch.shape = (x_shape[0], x_shape[1], window_num*sr*self.splited_window_size)
            
        


        type_1_len = len(type_1_seg)
        type_2_len = len(type_2_seg)
        type_3_len = len(type_3_seg)

        y_batch = np.concatenate(( np.ones(type_1_len), 
                                    np.zeros(type_2_len), 
                                    np.zeros(type_3_len)))

        y_batch = np.asarray(y_batch, dtype=np.int32)
        y_batch = self.iden_mat[y_batch]
        
     

        return x_batch, y_batch
    



class ViTGenerator_one_channel(Sequence):
    def __init__(self,type_1_data=[], type_2_data=[], type_3_data=[], batch_size=500, data_type = 'snu', ds_factor = 1, scale_resolution = 128, sampling_rate = 200):
        self.ratio_type_1 = [5,4,3,2]
        self.ratio_type_2 = [0.05,0.05,0.05,0.05]
        self.ratio_type_3 = [5,6,7,8]
        self.batch_size = batch_size
        self.epoch = 0
        self.update_period = 10
        self.type_1_data = np.array(type_1_data)
        self.type_2_data = np.array(type_2_data)
        self.type_3_data = np.array(type_3_data)
        self.type_1_len = len(type_1_data)
        self.type_2_len = len(type_2_data)
        self.type_3_len = len(type_3_data)
        self.data_type = data_type
        self.scale_resolution = scale_resolution
        self.sampling_rate = sampling_rate
        self.manual_channels = ['FP1-F7']
        #self.manual_channels = ['FP1-F7']
        self.ds_factor = ds_factor

        self.iden_mat = np.eye(2)

        self.batch_set, self.batch_num = updateDataSet( self.type_1_len,
                                                        self.type_2_len, 
                                                        self.type_3_len, 
                                                        [self.ratio_type_1[0], self.ratio_type_2[0],self.ratio_type_3[0]], 
                                                         int(self.batch_size/len(self.manual_channels)))
    def on_epoch_end(self):
        self.epoch += 1
        self.ratio_idx = 0
    def __len__(self):
        return self.batch_num
    
    def NotNoneBatch(self, idx):
        batch_concat = None
        if len(self.batch_set[1][idx]) == 0:
            type_1_seg = []
        else:
            type_1_seg = self.type_1_data[self.batch_set[0][self.batch_set[1][idx]]]
            batch_concat = type_1_seg
  
        if len(self.batch_set[3][idx]) == 0:
            type_2_seg = []
        else:
            type_2_seg = self.type_2_data[self.batch_set[2][self.batch_set[3][idx]]]
            batch_concat = np.concatenate((batch_concat, type_2_seg))
        if len(self.batch_set[5][idx]) == 0:
            type_3_seg = []
        else:
            type_3_seg = self.type_3_data[self.batch_set[4][self.batch_set[5][idx]]]
            batch_concat = np.concatenate((batch_concat, type_3_seg))

        return batch_concat, type_1_seg, type_2_seg, type_3_seg
    
    def __getitem__(self, idx):
        
        batch_concat, type_1_seg, type_2_seg, type_3_seg = self.NotNoneBatch(idx)
        raw_batch = Segments2Data(batch_concat, manual_channels=self.manual_channels)
        raw_batch.shape = (raw_batch.shape[0]*raw_batch.shape[1],raw_batch.shape[2])
        x_batch = np.array(PreProcessing.SegmentsCWT(raw_batch, sampling_rate = self.sampling_rate, scale_resolution = self.scale_resolution))
        x_batch = x_batch[:,:,::self.ds_factor]
                
        x_batch = np.expand_dims(x_batch, axis = -1)

        type_1_len = len(type_1_seg) * len(self.manual_channels)
        type_2_len = len(type_2_seg) * len(self.manual_channels)
        type_3_len = len(type_3_seg) * len(self.manual_channels)
       
        y_batch = np.concatenate(( np.ones(type_1_len )*1, 
                                   np.ones(type_2_len )*0, 
                                   np.ones(type_3_len )*0))
        y_batch = np.asarray(y_batch, dtype=np.int32)
        y_batch = self.iden_mat[y_batch]

        
        return x_batch, y_batch
    
class AutoEncoderGenerator(ViTGenerator_one_channel):
    def __init__(self,type_1_data=[], type_2_data=[], type_3_data=[], batch_size=500, data_type = 'snu'):
        super().__init__(type_1_data, type_2_data, type_3_data, batch_size, data_type)
        #self.manual_channels = ['FP1-F7']

    def on_epoch_end(self):
        self.batch_set, self.batch_num = updateDataSet( self.type_1_len,
                                                        self.type_2_len, 
                                                        self.type_3_len, 
                                                        [self.ratio_type_1[0], self.ratio_type_2[0],self.ratio_type_3[0]], 
                                                         int(self.batch_size/len(self.manual_channels)))

    def __getitem__(self, idx):
        batch_concat, type_1_seg, type_2_seg, type_3_seg = self.NotNoneBatch(idx)
        raw_batch = Segments2Data(batch_concat, type = self.data_type)
        #raw_batch.shape = (raw_batch.shape[0]*raw_batch.shape[1],1,raw_batch.shape[2])
        x_batch = np.expand_dims(raw_batch, axis = -1)
        #x_batch = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)(x_batch)
        x_batch = x_batch / 50
        

        return x_batch, x_batch
    
class Singal2FullGenerator(ViTGenerator_one_channel):
    def __init__(self, full_ch_model, type_1_data=[], type_2_data=[], type_3_data=[], batch_size=500, data_type = 'snu'):
        super().__init__(type_1_data, type_2_data, type_3_data, batch_size, data_type)
        self.full_ch_encoder = copy.deepcopy(full_ch_model)
        if data_type == 'chb_one_ch' : data_type = 'chb'
        if data_type == 'snu_one_ch' : data_type = 'snu'
        #self.manual_channels = ['FP1-F7']

    def on_epoch_end(self):
        self.batch_set, self.batch_num = updateDataSet( self.type_1_len,
                                                        self.type_2_len, 
                                                        self.type_3_len, 
                                                        [self.ratio_type_1[0], self.ratio_type_2[0],self.ratio_type_3[0]], 
                                                         int(self.batch_size/len(self.manual_channels)))

    def __getitem__(self, idx):
        batch_concat, type_1_seg, type_2_seg, type_3_seg = self.NotNoneBatch(idx)
        full_channel_batch = Segments2Data(batch_concat, type = self.data_type)
        
        
        #raw_batch.shape = (raw_batch.shape[0]*raw_batch.shape[1],1,raw_batch.shape[2])
        full_channel_batch = np.expand_dims(full_channel_batch, axis = -1) # (batch, 18, 1000, 1)
        full_channel_batch = full_channel_batch / 50
        single_channel_batch = full_channel_batch[:,0,:,:] # (batch, 1000, 1)

        x_batch = np.expand_dims(single_channel_batch, axis = -3)
        y_batch = self.full_ch_encoder.predict_on_batch(full_channel_batch)
        
        #x_batch = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)(x_batch)
        

        return x_batch, y_batch

class ViTGenerator_full_channel(Sequence):
    def __init__(self,type_1_data=[], type_2_data=[], type_3_data=[], batch_size=500, data_type = 'snu', scale_resolution = 128, sampling_rate = 200):
        self.ratio_type_1 = [5,4,3,2]
        self.ratio_type_2 = [0.05,0.05,0.05,0.05]
        self.ratio_type_3 = [5,6,7,8]
        self.batch_size = batch_size
        self.epoch = 0
        self.update_period = 10
        self.type_1_data = np.array(type_1_data)
        self.type_2_data = np.array(type_2_data)
        self.type_3_data = np.array(type_3_data)
        self.type_1_len = len(type_1_data)
        self.type_2_len = len(type_2_data)
        self.type_3_len = len(type_3_data)
        self.data_type = data_type
        self.scale_resolution = scale_resolution
        self.sampling_rate = sampling_rate

        self.iden_mat = np.eye(2)

        self.batch_set, self.batch_num = updateDataSet( self.type_1_len,
                                                        self.type_2_len, 
                                                        self.type_3_len, 
                                                        [self.ratio_type_1[0], self.ratio_type_2[0],self.ratio_type_3[0]], 
                                                        self.batch_size)
    def on_epoch_end(self):
        self.epoch += 1
        if self.epoch/self.update_period < 4:
            self.ratio_idx = int(self.epoch/self.update_period)
        else:
            self.ratio_idx = 3
        self.ratio_idx = 0
        self.batch_set, self.batch_num = updateDataSet (self.type_1_len,
                                                        self.type_2_len,
                                                        self.type_3_len, 
                                                        [self.ratio_type_1[self.ratio_idx], 
                                                         self.ratio_type_2[self.ratio_idx], 
                                                         self.ratio_type_3[self.ratio_idx]], 
                                                         self.batch_size)
    def __len__(self):
        return self.batch_num
    
    def __getitem__(self, idx):
        batch_concat = None
        if len(self.batch_set[1][idx]) == 0:
            type_1_seg = []
        else:
            type_1_seg = self.type_1_data[self.batch_set[0][self.batch_set[1][idx]]]
            batch_concat = type_1_seg
  
        if len(self.batch_set[3][idx]) == 0:
            type_2_seg = []
        else:
            type_2_seg = self.type_2_data[self.batch_set[2][self.batch_set[3][idx]]]
            batch_concat = np.concatenate((batch_concat, type_2_seg))
        if len(self.batch_set[5][idx]) == 0:
            type_3_seg = []
        else:
            type_3_seg = self.type_3_data[self.batch_set[4][self.batch_set[5][idx]]]
            batch_concat = np.concatenate((batch_concat, type_3_seg))
            
        raw_batch = Segments2Data(batch_concat, type='chb')
        batch_size = raw_batch.shape[0]
        channel_num = raw_batch.shape[1]
        raw_batch.shape = (raw_batch.shape[0]*raw_batch.shape[1],raw_batch.shape[2])
        x_batch = np.array(PreProcessing.SegmentsCWT(raw_batch, sampling_rate = self.sampling_rate, scale_resolution = self.scale_resolution))
        x_batch.shape = (batch_size, channel_num, x_batch.shape[-2], x_batch.shape[-1])        
        
        x_batch = np.expand_dims(x_batch, axis = -1)
        type_1_len = len(type_1_seg)
        type_2_len = len(type_2_seg)
        type_3_len = len(type_3_seg)
       
        y_batch = np.concatenate(( np.ones(type_1_len )*1, 
                                   np.ones(type_2_len )*0, 
                                   np.ones(type_3_len )*0))
        y_batch = np.asarray(y_batch, dtype=np.int32)
        y_batch = self.iden_mat[y_batch]

        
        return x_batch, y_batch
    


