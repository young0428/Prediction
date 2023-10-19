import pickle
import pyedflib
import numpy as np
import os



with pyedflib.EdfReader("/host/d/SNU_DATA/SNU001/SNU001.edf") as f:
    edf_signal = f.readSignal(1,0,20)
a = np.array(edf_signal)/1000
print(a)