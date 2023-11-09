import TrainAutoEncoder_paper
import TrainLSTM

if __name__ == "__main__":
    encoder_model_name = "paper_base_rawEEG_encoder_for_categorical"
    lstm_model_name = "paper_base_rawEEG_categorical"
    #TrainAutoEncoder_paper.train(encoder_model_name)
    TrainLSTM.train(lstm_model_name,encoder_model_name)