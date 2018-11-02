import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dir = 'Bootcamp/DataVisualization/'

df1 = pd.read_csv(dir + 'df1', index_col=0)
df2 = pd.read_csv(dir + 'df2')

sns.set_style('darkgrid')

df1['A'].plot.hist(bins=20)
plt.show()

df2.plot.area(alpha=0.4)
plt.show()

df2.plot.bar(alpha=0.4,stacked=True)
plt.show()

df1.plot.line(y='B', figsize=(8, 6), lw=4)
plt.show()

df1.plot.scatter(x='A', y='B', c='C',s=df1['C']*100, figsize=(20, 10), cmap='plasma')
plt.show()

df2.plot.box()
plt.show()

df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
df.plot.hexbin(x='a', y='b', gridsize=25, figsize=(6, 4), cmap='plasma')
plt.show()

df2['a'].plot.kde()
plt.show()