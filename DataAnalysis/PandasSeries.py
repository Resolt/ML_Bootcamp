import numpy as np
import pandas as pd

labels = ['a', 'b', 'c', 'd']
my_data = [10, 20, 30]
arr = np.array(my_data)
d = {'a': 10, 'b': 20, 'c': 30}

s1 = pd.Series(my_data, index=labels[:len(my_data)])
s2 = pd.Series(arr, index=labels[:len(my_data)])
s3 = pd.Series(d)
s4 = pd.Series(labels)

print(s1)
print(s2)
print(s3)
print(s4)

ser1 = pd.Series(
	data=[1, 2, 3, 4],
	index=["USA", "Japan", "USSR", "Germany"]
)

ser2 = pd.Series(
	data=[1, 2, 5, 4],
	index=["USA", "Japan", "Italy", "Germany"]
)

print(ser1['USA'])
print(s4[3])

print(ser1 + ser2)