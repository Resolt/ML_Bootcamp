import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn import metrics

plt.style.use('ggplot')
# plt.tight_layout()

dir = 'Bootcamp/LinearRegression/'
df = pd.read_csv(dir+'Ecommerce Customers')
print(df.head())
print(df.info())
print(df.describe())

# PLOTTING
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=df)

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df)

sns.jointplot(x='Time on App',y='Length of Membership',data=df,kind='hex')

sns.pairplot(df) # LENGTH OF MEMBERSHIP LOOKS LIKE THE MOST CORRELATED FEATURE

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent', data=df)

plt.show()

# LINEAR REGRESSION
cols = ['Avg. Session Length','Time on App','Time on Website','Length of Membership']
X = df[cols]
y = df['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LR(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
lm.fit(X_train, y_train)

# PRINT THE COEFFICIENTS NICELY IN A DATAFRAME
print("Coefficients of fitted model:\n{}\n".format(
	pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
))

# PUT THE TARGETS AND PREDICTIONS ON DATAFRAME AND PRINT CORRELATIONS
pred = pd.DataFrame({'A':y_test,'B':lm.predict(X_test)})
print("Correlation: {}\n".format(pred.corr()))

# SCATTER PLOT THE TARGETS AND THE PREDICTIONS
sns.scatterplot(x='A',y='B',data=pred)
plt.show()

# EVALUATE MODEL PERFORMANCE WITH THESE METRICS (LOWER IS BETTER)
print("MAE: {}\n".format(metrics.mean_absolute_error(pred['A'], pred['B'])))
print("MSE: {}\n".format(metrics.mean_squared_error(pred['A'], pred['B'])))
print("RMSE: {}\n".format(np.sqrt(metrics.mean_squared_error(pred['A'], pred['B']))))

# PLOT THE RESIDUALS (TARGET MINUS PREDICTION - DISTANCE TO TARGETS)
sns.distplot((pred['A']-pred['B']))
plt.show()

# THE REQUESTED DATA FRAME WAS CREATED EARLIER
# CONCLUSIONS
# A - Time on website has next to no influence on yearly amount spent
# B - Length of membership has most influence, so keeping people around appears to be the most important factor (loyalty plan?)
# C - Spending more time on mobile affects the spending as well, UXD should be of high concern