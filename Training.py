import TrainAutoEncoder_paper
import TrainLSTM
import TestFullModel
import pickle
import matplotlib.pyplot as plt
if __name__ == "__main__":
    encoder_model_name = "paper_base_encoder_128_chb_use_alldata"
    lstm_model_name = "paper_base_128ch_categorical_chb_binary_show_interictal"
    #%%
    #TrainAutoEncoder_paper.train(encoder_model_name,'chb')
    #%%
    TrainLSTM.train(lstm_model_name,encoder_model_name,'chb')
    #%%
    matrix,tf_matrix,sens,far = TestFullModel.validation(lstm_model_name,'chb',5,3)
    hist_list = [matrix, tf_matrix, sens, far]
    with open(f'./LSTM/{lstm_model_name}/MatrixHistory', 'wb') as file_pi:
        pickle.dump(hist_list, file_pi)

    # %%
    TestFullModel.SaveAsHeatmap(matrix, f"./LSTM/{lstm_model_name}/categorical_matrix.png")
    plt.clf()
    TestFullModel.SaveAsHeatmap(tf_matrix, f"./LSTM/{lstm_model_name}/tf_matrix.png" )

# %%
