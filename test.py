
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

window_size = 2
sr = 200
inputs = Input(shape=(1,int(window_size*sr)))
dilation_output = dilationnet(inputs)
model = Model(inputs = inputs, outputs = dilation_output)
model.summary()
# %%
