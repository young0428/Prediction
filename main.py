import ViTModelFunc

if __name__ == "__main__":
    model_name = "mobile_vit_one_channel_chb_4ch_dropout_transpose"
    ViTModelFunc.train(model_name, data_type = 'chb_one_ch')
