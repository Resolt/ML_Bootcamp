import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset('tips')
print(tips.head())

# plt.figure(figsize=(8, 6))
sns.set_style('darkgrid')
sns.set_context('notebook')
sns.despine()
sns.countplot(x='sex', data=tips)
# plt.tight_layout()
plt.show()

sns.lmplot(x='total_bill', y='tip', data=tips, hue='smoker',palette='plasma')
plt.show()
