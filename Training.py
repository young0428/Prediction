import TrainAutoEncoder_paper
import TrainLSTM
import TestFullModel
if __name__ == "__main__":
    encoder_model_name = "paper_base_encoder_64"
    lstm_model_name = "paper_base_64ch_categorical"
    #TrainAutoEncoder_paper.train(encoder_model_name,'snu')
    #%%
    TrainLSTM.train(lstm_model_name,encoder_model_name,'snu')
    #%%
    matrix,tf_matrix,sens,far = TestFullModel.validation(lstm_model_name,'snu')