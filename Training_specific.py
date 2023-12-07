#%%
import TrainLSTM_patient
if __name__ == "__main__":
    encoder_model_name = "autoencoder_for_snu_patient_specific"
    lstm_model_name = "patient_specific_chb_DCAE_LSTM_inter_gap_7200"
    #%%
    #TrainAutoEncoder_paper.train(encoder_model_name,'snu')
    #%%
    TrainLSTM_patient.train(lstm_model_name,encoder_model_name,'snu')
    #%%

# %%
