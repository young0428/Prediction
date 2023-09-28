import pyedflib

path = "E:/SNUH_START_END/patient_1/patient_1.edf"
f = pyedflib.EdfReader(path)

freq = f.getSampleFrequencies()
headers = f.getLabel(0)
num = f.getNSamples()

print(num/256)


