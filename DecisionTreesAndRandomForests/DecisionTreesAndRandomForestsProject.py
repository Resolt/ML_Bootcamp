import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split as TTS
from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.metrics import classification_report as CR
from sklearn.metrics import confusion_matrix as CM

plt.style.use('ggplot')

# READ THE DATA
dir="Bootcamp/DecisionTreesAndRandomForests/"
df = pd.read_csv(dir+'loan_data.csv')
print(df.info())
print(df.head())
print(df.describe())

# PLOTTING
# sns.distplot(df['fico'][df['credit.policy'] == 0],kde=False,bins=30)
# sns.distplot(df['fico'][df['credit.policy'] == 1],kde=False,bins=30)
# plt.show()

# sns.distplot(df['fico'][df['not.fully.paid'] == 0],kde=False,bins=30)
# sns.distplot(df['fico'][df['not.fully.paid'] == 1],kde=False,bins=30)
# plt.show()

# sns.countplot(x='purpose',data=df,hue='not.fully.paid')
# plt.show()

# sns.jointplot(x='fico',y='int.rate',data=df)
# plt.show()

# sns.lmplot(y='int.rate',x='fico',data=df,hue='credit.policy',col='not.fully.paid',palette='Set1')
# plt.show()

# DUMMI VARIABLES
final = pd.get_dummies(data=df,columns=['purpose'],drop_first=True)
print(final.head())

# SPLIT DATA
X_train,X_test,y_train,y_test = TTS(final.drop('not.fully.paid',axis=1),final['not.fully.paid'],test_size=0.3,random_state=101)

# DECISION TREE CLASSIFIER
tree = DTC()
tree.fit(X_train,y_train)

# PREDICT
tpred = tree.predict(X_test)

print(CR(y_test,tpred))
print(CM(y_test,tpred))

# RANDOM FOREST CLASSIFIER
forest = RFC(n_estimators=500)
forest.fit(X_train,y_train)
fpred = forest.predict(X_test)

print(CR(y_test,fpred))
print(CM(y_test,fpred))

# RANDOM FOREST PERFORMED BETTER OVER ALL - BUT THE FALSE NEGATIVES INCREASED COMAPRED TO A SINGLE TREE