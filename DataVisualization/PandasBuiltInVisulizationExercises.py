import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dir = 'Bootcamp/DataVisualization/'
df3 = pd.read_csv(dir + 'df3')

print(df3.info())
print(df3.head())

sns.set_style('darkgrid')

df3.plot.scatter(x='a', y='b', s=3, color='red', figsize=(12, 3), xlim=(-.2, 1.2), ylim=(-.2,1.2))
plt.show()

df3['a'].hist(color='blue',edgecolor='black')
plt.show()

plt.style.use('ggplot')

df3['a'].plot.hist(color='red', alpha=0.4, bins=25)
plt.show()

df3[['a', 'b']].plot.box()
plt.show()

df3['d'].plot.kde(lw=3,style="--")
plt.show()

df3.ix[np.arange(0, 30)].plot.area()
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
