
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import F1Score, Recall, Precision

import TestFullModel_specific
from readDataset import *
import AutoEncoder 
from LSTMmodel import LSTMLayer
from sklearn.model_selection import KFold
from ModelGenerator import FullModel_generator
from dilationmodel import *

window_size = 120
splited_window_size = 2
sampling_rate = 200


inputs = Input(shape=(1, window_size * sampling_rate))
ts_output = td_net(inputs, splited_window_size=splited_window_size, sampling_rate=sampling_rate)
model = Model(inputs,ts_output)
model.summary()

# %%
