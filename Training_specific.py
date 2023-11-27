#%%
import TrainAutoEncoder_paper
import TrainLSTM_patient
import TestFullModel
import pickle
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    encoder_model_name = "patient_specific_chb_autoencoder_no_resample_preictal"
    lstm_model_name = "patient_specific_chb_DCAE_LSTM_inter_gap_7200"
    #%%
    #TrainAutoEncoder_paper.train(encoder_model_name,'chb')
    #%%
    TrainLSTM_patient.train(lstm_model_name,encoder_model_name,'chb')
    #%%

# %%
