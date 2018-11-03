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

train = pd.read_csv(dir+'titanic_train.csv')
test = pd.read_csv(dir+'titanic_test.csv')

# EXPLORE VIA PLOTTING - PART 1

# sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
# sns.countplot(x='Survived',data=train,hue='Pclass')
# sns.distplot(train['Age'].dropna(),kde=False,bins=30)
# sns.countplot(x='SibSp',data=train)
# sns.distplot(train['Fare'],kde=False)

# CLEANING DATA - PART 2

# AGE

# FILL OUT AGE VIA LINEAR REGRESSION
# CREATE THE TRAINING SETS AND SET SEX TO BE 0 OR 1
X_train_age = train.dropna(subset=['Age']).drop(['Cabin', 'Age', 'Name', 'Ticket', 'Embarked'], axis=1)
X_train_age['Sex'] = X_train_age['Sex'].map({'male':0,'female':1})
y_train_age = train.dropna(subset=['Age'])['Age']
# PREPARE THE PREDICTION SET AND SET SEX TO BE 0 OR 1
X_pred_age = train[np.invert(train.index.isin(X_train_age.index))].drop(['Cabin', 'Age', 'Name', 'Ticket', 'Embarked'], axis=1)
X_pred_age['Sex'] = X_pred_age['Sex'].map({'male':0,'female':1})

# CREATE AND FIT THE MODEL
lm = LR()
lm.fit(X_train_age,y_train_age)

# PREDICT AGES AND INSERT
train.loc[np.isnan(train['Age']), 'Age'] = lm.predict(X_pred_age)

# IMPUTANCE
def impute_age(cols):
	Age = cols[0]
	Pclass = cols[1]

	if pd.isnull(Age):
		if Pclass == 1:
			return 37
		elif Pclass == 2:
			return 29
		else:
			return 24

	else:
		return Age

age_imp = train[['Age','Pclass']].apply(impute_age,axis=1)

# sns.scatterplot(x=age_imp,y=train['Age']) # LOOKS VERY MUCH THE SAME, BUT I'D RATHER USE THE REGRESSION DATA
# plt.show()

# CABIN - TOO MUCH MISSING DATA

train.drop('Cabin',axis=1,inplace=True)

# WE NOW ONLY HAVE A SINGLE DATAPOINT IN EMBARK BEING NAN SO LET'S YOLO THE REST

train.dropna(inplace=True)

# DUMMY VARIABLES (FACTOR SPLIT TO BINARY)
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
pclass = pd.get_dummies(train['Pclass']) # PCLASS IS ALREADY NUMERIC BUT IN INTEGERS REPRESENTING FACTORS - PRECISION IS SLIGHT INCREASED BY SPLITTING TO BINARY COLUMNS

train = pd.concat([train,sex,embark], axis=1)
train = pd.concat([train,pclass],axis=1) # IN SEPARATE LINE FOR EASY ON/OFF SWITCH

# DROP THE NO LONGE NEEDED COLUMNS
train.drop(['Sex', 'Embarked', 'Name', 'Ticket','PassengerId'],inplace=True,axis=1)
train.drop(['Pclass'],inplace=True,axis=1) # IN SEPARATE LINE FOR EASY ON/OFF SWITCH

# PART 3 - TRAINING AND PREDICTION
X = train.drop('Survived',axis=1)
y = train['Survived']

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=101)

lg = LGR()
lg.fit(X_train,y_train)

# EVALUATE THE RESULTS
pred = pd.DataFrame({'P':lg.predict(X_test),'R':y_test})

print(pred.corr())
print(CR(pred['R'],pred['P']))
print(CM(pred['R'],pred['P']))

