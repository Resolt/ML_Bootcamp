import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split as TTS

from sklearn.datasets import load_breast_cancer

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV as GSCV

from sklearn.metrics import classification_report as CR, confusion_matrix as CM
# from sklearn.metrics import confusion_matrix as CM

plt.style.use('ggplot')

# READ THE DATA
cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer['data'],columns=cancer['feature_names'])
print(df.info())

# EXPLORATION
# sns.pairplot(df)
# plt.show()

# SPLIT
X_train,X_test,y_train,y_test = TTS(df,cancer['target'],test_size=0.3,random_state=64)

# MODEL FITTING
model = SVC()
model.fit(X_train,y_train)
pred = model.predict(X_test)

print(CR(y_test,pred))
print(CM(y_test,pred))
# VERY BAD!!!

# GRID SEARCH - TRIAL AND ERROR ON VARIOUS MODEL PARAMTERS
param_grid = {'C':list(np.arange(50000,500000,10000)),'gamma':list(np.arange(0.00000001,0.000005,0.0000005))}

grid = GSCV(SVC(),param_grid,verbose=3,n_jobs=4)
grid.fit(X_train,y_train)

print(grid.best_params_)
print(grid.best_estimator_)

gpred = grid.predict(X_test)
print(CR(y_test,gpred))
print(CM(y_test,gpred))