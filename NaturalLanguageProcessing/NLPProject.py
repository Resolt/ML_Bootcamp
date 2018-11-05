import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split as TTS

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfTransformer as TT

from sklearn.naive_bayes import MultinomialNB as MNB

from sklearn.metrics import classification_report as CR

plt.style.use('ggplot')

# READ THE DATA
dir="Bootcamp/NaturalLanguageProcessing/"
df = pd.read_csv(dir+'yelp.csv')
print(df.head())
print(df.info())
print(df.describe())

# CREATE TEXT LENGTH COLUMN
df['text_length'] = df['text'].apply(lambda x: len(x.split()))

# EXPLORATION

# PLOTTING
# g = sns.FacetGrid(data=df,col='stars',height=5)
# g.map(plt.hist, 'text_length')
# plt.show()

# sns.boxplot(x='stars',y='text_length',data=df)
# plt.show()

# sns.countplot(x='stars',data=df)
# plt.show()

# GROUPING AND CORRELATION
print(df.groupby('stars').mean().head(5))
print(df.groupby('stars').mean().head(5).corr())

# sns.heatmap(df.groupby('stars').mean().head(5).corr(), annot=True)
# plt.show()

# CREATE YELP DF ONLY FOR 1 AND 5 STAR REVIEWS
yelp_class = df[(df['stars'] == 1) | (df['stars'] == 5)]

X = yelp_class['text']
y = yelp_class['stars']

# CREATE COUNT VECTORIZER AND FIT TO X
cv = CV().fit(X)

# OVERWRITE X WITH TRANSFORM
X = cv.transform(X)

# TRAIN TEST SPLIT
X_train,X_test,y_train,y_test = TTS(X,y,test_size=0.3,random_state=64)

# CREATE NAIVE BAYES OBJECT AND FIT
nb = MNB().fit(X_train,y_train)
pred = nb.predict(X_test)

print(CR(y_test,pred))

# PIPELINE
pipe = Pipeline([
	('CV', CV()),
	('TFIDF', TT()),
	("BAYES", MNB())
])

# REDO SPLIT
X = yelp_class['text']
y = yelp_class['stars']
X_train,X_test,y_train,y_test = TTS(X,y,test_size=0.3,random_state=64)

# FIT THE PIPE
pipe.fit(X_train,y_train)

# PREDICT WITH PIPE
pred_pipe = pipe.predict(X_test)

print(CR(y_test, pred_pipe))

# TFIDF MADE THINS CONSIDERABLY WORSE