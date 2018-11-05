import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import StandardScaler as SS

from sklearn.decomposition import PCA

plt.style.use('ggplot')

# GET THE CANCER DATA
cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df.info())

# SCALE THE DATA
scaler = SS()
scaler.fit(df)
scaled_data = scaler.transform(df)

# PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

# K MEANS - MAKES A LOT OF SENSE TO APPLY THIS ON TOP OF PCA

from sklearn.cluster import KMeans as KM

model = KM(n_clusters=2)
model.fit(df)
fig,axes = plt.subplots(1,2,figsize=(10,6))
fig.suptitle('Breast Cancer', fontsize=20)

axes[0].set_title('Diagnosis')
axes[0].scatter(x_pca[:,0],x_pca[:,1],c=model.labels_,cmap='coolwarm',)
axes[1].set_title('KMC')
axes[1].scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='coolwarm')

plt.show()

# HEAT MAP - JUST FOR PCA
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])

sns.heatmap(df_comp,cmap='coolwarm')
plt.show()
