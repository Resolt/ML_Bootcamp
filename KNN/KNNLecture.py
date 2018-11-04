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
df = pd.read_csv(dir+'Classified Data',index_col=0)

# INSTANTIATE SCALER - THIS IS FOR KIND OF NORMLIZATION OF THE DATA IN THE DATAFRAME
scaler = SS()
scaler.fit(df.drop('TARGET CLASS',axis=1))

# DO THE SCALING
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

# PUT RESULTS IN DATASET WITH CORRESPONDING COLUMN NAMES
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])

# SPLIT THE DATA
X_train,X_test,y_train,y_test = TTS(df_feat,df['TARGET CLASS'],test_size=0.3,random_state=101)

# INSTANTIATE MODEL AND FIT
model = KNC(n_neighbors=1)
model.fit(X_train,y_train)

# PREDICT
pred = model.predict(X_test)

# PERFORMANCE
print(CR(y_test,pred))
print(CM(y_test,pred))

# EMPTY LIST FOR CONTAINING THE PERFORMANCE RESULTS
error_rate = []

# ELBOW METHOD FOR CHOSING A BETTER K VALUE (ITERATE THROUGH A RANGE OF K VALUES AND SEE WHICH ONE IS BEST)
for i in range(1,40):
	model = KNC(n_neighbors=i)
	model.fit(X_train,y_train)
	pred_i = model.predict(X_test)
	error_rate.append(np.mean(pred_i != y_test))


sns.lineplot(x=np.arange(1,40),y=np.array(error_rate))
plt.show()

# LOOKS LIKE 17 WOULD BE A GOOD BET
