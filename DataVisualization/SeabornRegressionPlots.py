import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset('tips')

sns.lmplot(x='total_bill', y='tip', data=tips, col='sex',hue='smoker',markers=['o', 'v'], scatter_kws={'s':100},aspect=0.6,size=8)
plt.tight_layout()
plt.show()

