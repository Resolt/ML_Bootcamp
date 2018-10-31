import numpy as np
import pandas as pd

from numpy.random import randn

np.random.seed(101)

# PART 1

df = pd.DataFrame(randn(4, 5), index=['W', 'X', 'Y', 'Z'], columns=['A', 'B', 'C', 'D', 'E'])

print(df)
print(df[['A', "B"]])

df['new'] = df['A'] + df['B']
print(df)

print(df.drop('new', axis=1))
print(df)
df.drop('new', axis=1, inplace=True)
print(df)

print(df['A'].iloc[2])

print(df.loc['Y', 'A'])

# PART 2

print(df > 0)
print(df[df['B'] < 0])
print(df[df['B'] < 0]['A'])
print(df[df['B'] < 0][['A', 'D']])
print(df['D'][(df['B'] < 0) & (df['A'] > 0)])
print(df['D'][(df['B'] < 0) | (df['A'] > 0)])

df.reset_index(inplace=True, drop=True)
print(df)
print(df['A'][df['A'] > 0])
print(df['A'][df['A'] > 0].iloc[1])

newind = 'CA NY WY OR '.split()
print(newind)

df['States'] = newind
print(df)

df.set_index('States', drop=True, inplace=True)
print(df)

# PART 3

outside = ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']
inside = [1, 2, 3, 1, 2, 3]
hier_index = list(zip(outside, inside))
print(hier_index)
hier_index = pd.MultiIndex.from_tuples(hier_index)
print(hier_index)

df = pd.DataFrame(randn(6, 2), hier_index, ['A', 'B'])
print(df)

print(df.loc['G1'].loc[1])
df.index.names = ['Groups', 'Numbers']

print(df)
print(df.xs(1, level='Numbers').xs('G1')['A'])