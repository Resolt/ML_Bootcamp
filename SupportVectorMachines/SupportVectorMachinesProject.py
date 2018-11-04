import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split as TTS

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV as GSCV

from sklearn.metrics import classification_report as CR, confusion_matrix as CM

plt.style.use('ggplot')

# GET THE DATA
iris = sns.load_dataset('iris')
print(iris.info())

# PLOTTING
# sns.pairplot(iris,hue='species')
# plt.show()

# sns.kdeplot(iris[['sepal_width','sepal_length']][iris['species'] == "setosa"])
# plt.show()

# SPLIT DATA
X_train,X_test,y_train,y_test = TTS(iris.drop('species',axis=1),iris['species'],test_size=0.3,random_state=101)

# TRAIN MODEL
model = SVC()
model.fit(X_train,y_train)
pred = model.predict(X_test)

print(CR(y_test,pred),CM(y_test,pred))
print(model)

# GRID SEARCH - "THIS IS NOT NECESSARY, THE MODEL IS PERFECT"
param_grid = {'C':list(np.arange(0.1,10,0.1)),'gamma':[1,0.1,0.001,0.0001]}

grid = GSCV(SVC(),param_grid,verbose=3,n_jobs=4)
grid.fit(X_train,y_train)

print(grid.best_params_)
print(grid.best_estimator_)

gpred = grid.predict(X_test)
print(CR(y_test,gpred))
print(CM(y_test,gpred))