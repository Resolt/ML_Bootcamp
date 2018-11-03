import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LogisticRegression as LGR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as CR
from sklearn.metrics import confusion_matrix as CM

plt.style.use('ggplot')

dir = 'Bootcamp/LogisticRegression/'

# READ DATA AND CHECK
ad_data = pd.read_csv(dir+'advertising.csv')
print(ad_data.head())
print(ad_data.info())
print(ad_data.describe())

# PLOTS
# ad_data['Age'].hist(bins=30)
# plt.show()

# sns.jointplot(x='Age',y='Area Income',data=ad_data)
# plt.show()

# sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde')
# plt.show()

# sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data)
# plt.show()

# sns.pairplot(ad_data,hue='Clicked on Ad')
# plt.show()

# CLEAN DATA
ad_data['Timestamp'] = pd.to_datetime(ad_data['Timestamp'])
ad_data['Hour'] = ad_data['Timestamp'].apply(lambda x: x.hour)

ad_data.drop(['Timestamp','Ad Topic Line','Country', 'City'],inplace=True,axis=1)

print(ad_data.info())

sns.heatmap(ad_data.isnull())
plt.show()

# REGRESSION
pname = 'Clicked on Ad'
X = ad_data.drop(pname,axis=1)
y = ad_data[pname]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lg = LGR()
lg.fit(X_train,y_train)
pred = lg.predict(X_test)

print(CR(y_test,pred))
print(CM(y_test,pred))