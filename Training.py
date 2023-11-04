import TrainAutoEncoder
import TrainLSTM

if __name__ == "__main__":
    encoder_model_name = "0.1_50_BandPass"
    lstm_model_name = "0.1_50_BandPass_model"
    TrainAutoEncoder.train(encoder_model_name)
    TrainLSTM.train(lstm_model_name,encoder_model_name)