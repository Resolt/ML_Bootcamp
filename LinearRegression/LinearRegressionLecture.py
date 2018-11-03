import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn import metrics

plt.style.use('ggplot')
plt.tight_layout()

dir = 'Bootcamp/LinearRegression/'

# READ HOUSING
df = pd.read_csv(dir+'USA_Housing.csv')
print(df.head())
print(df.info())
print(df.describe())

# PLOT
# sns.distplot(df['Price'])
# plt.show()

# sns.heatmap(df.corr(),annot=True)
# plt.show()

# REGERESSION
X = df[df.columns[range(5)]]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LR()
lm.fit(X_train, y_train)

print("Intecept: {}\n".format(lm.intercept_))
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
print("Coefficients:\n{}\n".format(cdf))

pred = pd.DataFrame({'A':lm.predict(X_test),'B':y_test})
# print(pred)
print("Correlation:\n{}\n".format(pred.corr()))

# BOSTON
from sklearn.datasets import load_boston
boston = load_boston()

dfb = pd.DataFrame(boston['data'], columns=boston['feature_names'])

t = boston['target']
dfb_train, dfb_test, t_train, t_test = train_test_split(dfb, t, test_size=0.3, random_state = 64)
blm = LR()
blm.fit(dfb_train, t_train)

bpred = pd.DataFrame({'A':blm.predict(dfb_test),'B':t_test})
bcdf = pd.DataFrame(blm.coef_,dfb.columns,columns=['Coeff'])
print(bcdf)
print(bpred.corr())

sns.scatterplot(data=bpred,x='A',y='B')
plt.show()

sns.distplot((t_test-bpred['A']))
plt.show()

print("MAE: {}\n".format(metrics.mean_absolute_error(t_test, bpred['A'])))
print("MSE: {}\n".format(metrics.mean_squared_error(t_test, bpred['A'])))
print("RMSE: {}\n".format(np.sqrt(metrics.mean_squared_error(t_test, bpred['A']))))