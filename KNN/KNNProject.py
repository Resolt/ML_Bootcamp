import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split as TTS

from sklearn.neighbors import KNeighborsClassifier as KNC

from sklearn.metrics import classification_report as CR
from sklearn.metrics import confusion_matrix as CM

plt.style.use('ggplot')

# READ THE DATA
dir="Bootcamp/KNN/"
df = pd.read_csv(dir+'KNN_Project_Data',index_col=0)
print(df.head())

# PLOTTING
# sns.pairplot(df)
# plt.show()

# SCALING
scaler = SS()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_scale = pd.DataFrame(scaled,columns=df.columns[:-1])
print(df_scale.head())

# SPLIT DATA INTO TRAINING AND TESTING
X_train,X_test,y_train,y_test = TTS(df_scale,df['TARGET CLASS'],test_size=0.3,random_state=101)

# KNN
model = KNC(n_neighbors=1)
model.fit(X_train,y_train)
pred = model.predict(X_test)

print(CR(y_test,pred))
print(CM(y_test,pred))

# CHOOSE K VALUE (ELBOW METHOD)
error_rate = []

for i in range(1,40):
	model = KNC(n_neighbors=i)
	model.fit(X_train,y_train)
	pred_i = model.predict(X_test)
	error_rate.append(np.mean(y_test != pred_i))

sns.lineplot(x=np.arange(1,40),y=np.array(error_rate))
plt.show()

# RERUN WITH NEW K
model = KNC(n_neighbors=37)
model.fit(X_train,y_train)
pred = model.predict(X_test)

print(CR(y_test,pred))
print(CM(y_test,pred))