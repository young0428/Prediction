import TrainAutoEncoder_paper
import TrainLSTM
import TestFullModel
if __name__ == "__main__":
    encoder_model_name = "paper_base_encoder_128"
    lstm_model_name = "paper_base_128ch_categorical"
    TrainAutoEncoder_paper.train(encoder_model_name)
    TrainLSTM.train(lstm_model_name,encoder_model_name)
    TestFullModel.validation(lstm_model_name)