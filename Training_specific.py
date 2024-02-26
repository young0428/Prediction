#%%

import ssqueezepy as ssq
import os
import logging
logging.getLogger('cupy').setLevel(logging.ERROR)
os.environ['SSQ_PARALLEL'] = '1'
os.environ['SSQ_GPU'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='cupy.cuda.compiler')

import TrainLSTM_patient
import TrainAutoEncoder_paper
import Train_Dilated
import Train_WCL
import Train_ViT
import gc
import multiprocessing as mp


if __name__ == "__main__":
    
    # encoder_model_name = "autoencoder_for_snu_patient_62_32_no_last_activation"
    # lstm_model_name = "one_channel_patient_specific_snu_DCAE_LSTM_62_32"
    # data_type = 'snu_one_ch'
    # # lstm_model_name = "one_channel_patient_specific_chb_DCAE_LSTM"
    data_type = 'snu_one_ch'

    dilation_model_name = 'snu_one_ch_dilation_lstm_30sec_random_filtering_ip21'
    #dilation_model_name = 'one_ch_mvit_6_4sec'
    #channels = ['FP1-F7','T7-P7','FP2-F4','T8-P8','P7-O1','P8-O2']
    channels = ["Fp1-AVG", "Fp2-AVG", "T3-AVG", "T4-AVG", "O1-AVG", "O2-AVG"]
    

    # try:
    #     mp.set_start_method('spawn')
    # except RuntimeError:
    #     pass
    channel_cnt_file_path = f"./Dilation/ch_cnt"
    path_dir = os.path.dirname(channel_cnt_file_path)
    
    if os.path.exists(channel_cnt_file_path):
        with open(channel_cnt_file_path,'r') as f:
            channel_cnt = int(f.read())
    else:
        
        with open(channel_cnt_file_path,'w') as f:
            channel_cnt = 0
            f.write('0')
            
    channel = channels[channel_cnt%len(channels)]
    model_name = dilation_model_name + '_' + channel
    channel_cnt += 1
    print(channel)
    with open(channel_cnt_file_path,'w') as f:
        f.write(str(channel_cnt))
    
    try:
        Train_Dilated.train(model_name,'',channel=[channel],data_type=data_type)
        #Train_ViT.train(model_name,'',channel=[channel],data_type=data_type)
        
        #Train_WCL.train(model_name,'',channel=[channel],data_type=data_type)
    except Exception as e:
        print(f"예외 발생: {e}")
        gc.collect()
        quit()
    else:
        gc.collect()
        quit()
        
        
            
            
    #%%

# %%
