#%%
import TrainLSTM_patient
import TrainAutoEncoder_paper
import Train_Dilated
import gc
if __name__ == "__main__":
    
    # encoder_model_name = "autoencoder_for_snu_patient_62_32_no_last_activation"
    # lstm_model_name = "one_channel_patient_specific_snu_DCAE_LSTM_62_32"
    # data_type = 'snu_one_ch'
    # # lstm_model_name = "one_channel_patient_specific_chb_DCAE_LSTM"
    data_type = 'chb_one_ch'

    dilation_model_name = 'one_ch_dilation_lstm_300sec_random'
    channels = ['FP1-F7','T7-P7','FP2-F4','T8-P8','P7-O1','P8-O2']
    
    # #TrainAutoEncoder_paper.train(encoder_model_name, data_type)
    # TrainLSTM_patient.train(lstm_model_name,encoder_model_name,data_type)

    # encoder_model_name = "single_ch_ae_for_chb_patient_62_32"
    # full_ch_ae_model_name = "autoencoder_for_chb_patient_62_32"
    # data_type = 'chb_one_ch'
    # #%%
    # TrainAutoEncoder_paper.train(encoder_model_name, data_type, full_ch_ae_model_name)
    # #%%
    # #TrainLSTM_patient.train(lstm_model_name,encoder_model_name,data_type)
    
    for channel in channels:
        model_name = dilation_model_name + '_' + channel
        try:
            Train_Dilated.train(model_name,'',channel=[channel],data_type=data_type)
        except Exception as e:
            print(f"예외 발생: {e}")
        finally:
            # 가비지 컬렉션 수행
            gc.collect()
            print("리소스 정리 및 가비지 컬렉션 완료")
            quit
    #%%

# %%
