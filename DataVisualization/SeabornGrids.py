import matplotlib.pylab as plt
import seaborn as sns

iris = sns.load_dataset('iris')
tips = sns.load_dataset('tips')
print(iris.head())
print(tips.head())

g = sns.PairGrid(iris)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
plt.show()

g = sns.FacetGrid(data=tips, col='time', row='smoker')
g.map(plt.scatter, 'total_bill', 'tip')
plt.show()