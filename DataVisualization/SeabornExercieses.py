import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
titanic = sns.load_dataset('titanic')
print(titanic.head())

sns.jointplot(x='fare', y='age', data=titanic)
plt.show()

sns.distplot(titanic['fare'], kde=False, bins=30)
plt.show()

sns.boxplot(x='class', y='age', data=titanic, palette='plasma')
plt.show()

sns.swarmplot(x='class', y='age', data=titanic, palette='plasma')
plt.show()

sns.countplot(titanic['sex'])
plt.show()

sns.heatmap(titanic.corr(), cmap='plasma')
plt.show()

g = sns.FacetGrid(data=titanic, col='sex')
g.map(plt.hist, 'age')
plt.show()
