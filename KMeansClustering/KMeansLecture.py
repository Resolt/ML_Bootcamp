import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs # FOR GENERATING BLOB DATA

from sklearn.cluster import KMeans as KM

plt.style.use('ggplot')

# MAKE DATA
data = make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.8,random_state=101)
df = pd.DataFrame(data[0])
print(df.info())

# PLOT
# plt.scatter(df[0],df[1],c=data[1])
# plt.show()

model = KM(n_clusters=4)
model.fit(df)
fig,axes = plt.subplots(1,2,figsize=(10,6))
fig.suptitle('K Means Clustering', fontsize=20)

axes[0].set_title('Unsupervised Algorithm Lables')
axes[0].scatter(df[0],df[1],c=model.labels_)
axes[1].set_title('Generator Labels')
axes[1].scatter(df[0],df[1],c=data[1])

plt.show()

