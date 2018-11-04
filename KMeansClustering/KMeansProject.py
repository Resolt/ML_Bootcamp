import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs # FOR GENERATING BLOB DATA

from sklearn.cluster import KMeans as KM

from sklearn.metrics import confusion_matrix as CM, classification_report as CR

plt.style.use('ggplot')

# READ THE DATA
dir="Bootcamp/KMeansClustering/"
df = pd.read_csv(dir+'College_Data',index_col=0)
print(df.info())
print(df.head())

# PLOTTING
# sns.scatterplot(x='Room.Board',y='Grad.Rate',hue='Private',data=df)
# plt.show()

# sns.scatterplot(x='Outstate',y='F.Undergrad',hue='Private',data=df)
# plt.show()

# sns.distplot(df['Outstate'][df['Private'] == 'No'],bins=30,kde=False)
# sns.distplot(df['Outstate'][df['Private'] == 'Yes'],bins=30,kde=False)
# plt.show()

# sns.distplot(df['Grad.Rate'][df['Private'] == 'No'],bins=30,kde=False)
# sns.distplot(df['Grad.Rate'][df['Private'] == 'Yes'],bins=30,kde=False)
# plt.show()

# INFO
print("Name of private school with grad rate higher than 100: {}\n".format(
	df.loc[df['Grad.Rate'] > 100].index.values[0]
))

df.loc[df['Grad.Rate'] > 100,'Grad.Rate'] = 100

# sns.distplot(df['Grad.Rate'][df['Private'] == 'No'],bins=30,kde=False)
# sns.distplot(df['Grad.Rate'][df['Private'] == 'Yes'],bins=30,kde=False)
# plt.show()

# K MEANS
# FIT
model = KM(n_clusters=2)
model.fit(df.drop('Private',axis=1))

# CENTER VECTORS
print("Fitted models center vectors:\n{}\n".format(
	model.cluster_centers_
))

# CREATE NUMERICAL CLUSTER COLUMN
df['Cluster'] = df['Private'].map({'Yes':1,'No':0})

print(CR(df['Cluster'],model.labels_))
print(CM(df['Cluster'],model.labels_))
