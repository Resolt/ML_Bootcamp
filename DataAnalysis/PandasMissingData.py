import numpy as np
import pandas as pd

d = {'A': np.array([1, 2, 3]), 'B': np.array([4, np.nan, np.nan]), 'C': np.array([1, np.nan])}
d['C'] = np.append(d['C'], 5)
# d['C'].append('WHAT')
# d['C'] = d['C'].astype('Float32')
print(d)

df = pd.DataFrame(d)
print(df)

print("C type: {}".format(df['C'].dtype))

print(df.dropna())
print(df.dropna(axis=0))
print(df.dropna(axis=1))

print(df.dropna(thresh=2))
print(df.dropna(axis = 1, thresh=2))

print(df.fillna(value="FILL"))
print(df.fillna(value=df.mean()))