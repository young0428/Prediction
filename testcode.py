import pyedflib
import natsort
import os

data_path = "D:/SNU_DATA"

file_list = natsort.natsorted(os.listdir(data_path))

for file_name in file_list:
    file_path = data_path+'/'+file_name+'/'+file_name+'.edf'
    num = int(file_name)
    os.rename(file_path,data_path+'/'+file_name+'/SNU%03d.edf'%(num))
    os.rename(data_path+'/'+file_name, data_path+'/SNU%03d'%(num))



