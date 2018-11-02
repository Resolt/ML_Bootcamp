import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset('tips')
print(tips.head())

# sns.distplot(tips['total_bill'], kde=True, bins=30)
sns.jointplot(x='total_bill',y='tip',data=tips, kind='hex')
# sns.pairplot(tips, hue='sex', palette=sns.xkcd_palette(sns.xkcd_rgb))
# sns.rugplot(tips['total_bill'])

# plt.tight_layout()
plt.show()

sns.kdeplot(tips['total_bill'])
plt.show()

