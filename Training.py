import TrainAutoEncoder_paper
import TrainLSTM
import TestFullModel
import pickle
import matplotlib.pyplot as plt
if __name__ == "__main__":
    encoder_model_name = "paper_base_encoder_128_chb_use_alldata"
    lstm_model_name = "patient_specific_chb_DCAE_LSTM"
    #%%
    #TrainAutoEncoder_paper.train(encoder_model_name,'chb')
    #%%
    TrainLSTM.train(lstm_model_name,encoder_model_name,'chb')
    #%%

# %%
