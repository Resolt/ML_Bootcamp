import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')

tc = tips.corr()

print(tc)

sns.heatmap(tc, annot=True, cmap='Oranges')
plt.show()

fp = flights.pivot_table(index='month', columns='year', values='passengers')

print(fp)

sns.heatmap(fp)
plt.tight_layout()
plt.show()

sns.clustermap(fp,cmap='coolwarm',standard_scale=1)
plt.show()