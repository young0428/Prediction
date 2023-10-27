import numpy as np

class test :
    def __init__(self, cnt):
        self.count = cnt
        self.count[0] += 1


a = [0]
b = test(a)
c = test(a)

print(a)

