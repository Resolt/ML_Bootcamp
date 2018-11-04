import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split as TTS
from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.metrics import classification_report as CR
from sklearn.metrics import confusion_matrix as CM

plt.style.use('ggplot')

# READ THE DATA
dir="Bootcamp/DecisionTreesAndRandomForests/"
df = pd.read_csv(dir+'kyphosis.csv')
print(df.head())
print(df.info())

# sns.pairplot(df,hue='Kyphosis')
# plt.show()

# SELECT X AND Y
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']

# SPLIT
X_train,X_test,y_train,y_test = TTS(X,y,test_size=0.3,random_state=101)

# DECISION TREE

# INSTANTIATE DECISION TREE CLASSIFIER
dtree = DTC()
# FIT MODEL
dtree.fit(X_train,y_train)
# GET PREDICTIONS
pred = dtree.predict(X_test)
# PERFORMANCE
print(CR(y_test,pred))
print(CM(y_test,pred))

# RANDOM FOREST (ENSEMBLE OF DECISION TREES)

# INSTANTIATE RANDOM FOREST CLASSIFIER
forest = RFC(n_estimators=200)
# FIT
forest.fit(X_train,y_train)
# PREDICTIONS
fpred = forest.predict(X_test)
# PERFORMANCE
print(CR(y_test,fpred))
print(CM(y_test,fpred))

# DECISION TREE IS BETTER, BUT NOT BY MUCH
# THIS MIGHT BE BEACUSE THE DATA SAMPLE IS SMALL AND THE TARGET VALUES IS SKEWED