#%%
import TrainLSTM_patient
import TrainAutoEncoder_paper
if __name__ == "__main__":
    encoder_model_name = "autoencoder_for_chb_patient_62_32"
    #encoder_model_name = "full_ch_autoencoder_for_chb_62_16"
    #lstm_model_name = "one_channel_patient_specific_chb_DCAE_LSTM"
    #data_type = 'chb_one_ch'
    data_type = 'chb'
    #%%
    TrainAutoEncoder_paper.train(encoder_model_name,data_type)
    #%%
    #TrainLSTM_patient.train(lstm_model_name,encoder_model_name,data_type)
    #%%

# %%
