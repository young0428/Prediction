#%%
import TrainLSTM_patient
import TrainAutoEncoder_paper
import Train_Dilated
if __name__ == "__main__":
    # encoder_model_name = "autoencoder_for_snu_patient_62_32_no_last_activation"
    # lstm_model_name = "one_channel_patient_specific_snu_DCAE_LSTM_62_32"
    # data_type = 'snu_one_ch'
    # # lstm_model_name = "one_channel_patient_specific_chb_DCAE_LSTM"
    data_type = 'chb_one_ch'

    dilation_model_name = 'one_ch_chb_dilation_model'
    Train_Dilated.train(dilation_model_name,'',data_type)
    
    # #TrainAutoEncoder_paper.train(encoder_model_name, data_type)
    # TrainLSTM_patient.train(lstm_model_name,encoder_model_name,data_type)

    # encoder_model_name = "single_ch_ae_for_chb_patient_62_32"
    # full_ch_ae_model_name = "autoencoder_for_chb_patient_62_32"
    # data_type = 'chb_one_ch'
    # #%%
    # TrainAutoEncoder_paper.train(encoder_model_name, data_type, full_ch_ae_model_name)
    # #%%
    # #TrainLSTM_patient.train(lstm_model_name,encoder_model_name,data_type)
    #%%

# %%
