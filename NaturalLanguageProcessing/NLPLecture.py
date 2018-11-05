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

import string

plt.style.use('ggplot')

dir="Bootcamp/NaturalLanguageProcessing/"

df = pd.read_csv(dir+'smsspamcollection/SMSSpamCollection',sep='\t',names=['label','msg'])

print(df.head())
print(df.describe())

# PART 1 - EXPLORE
print(df.groupby('label').describe())

# CREATE COLUMN FOR MESSAGE LENGTH
df['length'] = df['msg'].apply(len)

# sns.distplot(df['length'])
# plt.show()

# PRINT THE LONGEST MESSAGE
print(df[df['length'] == 910]['msg'].iloc[0])

# EVALUATE MESSAGE LENGTH BY SPAM/HAM
# df.hist(column='length',by='label',bins=30,figsize=(12,6))
# plt.show()

# PART 2 - CLEANING THE TEXT
# EXAMPLE MESSAGE WITH PUNCTUATION
mess = "Sample message! Notice: it has punctuation."

# REMMOVE PUNCTUATION AND PUT IN LIST
nopunc = [c for c in mess if c not in string.punctuation]
# AND JOIN TO STRING
nopunc = ''.join(nopunc)
# AND PRINT
print(nopunc)

# STOP WORDS - WORDS ONE WOULD COMMONLY REMOVE
sw = stopwords.words('english')

# CLEAN UP MESS
clean_mess = [word for word in nopunc.split() if word.lower() not in sw]
print(clean_mess)

# CREATE FUNCTION FOR CLEANING UP MESSAGES
def text_process(mess):
	np = ''.join([c for c in mess if c not in string.punctuation])
	return [word.lower() for word in np.split() if word.lower() not in sw]


# EXAMPLE CLEAN OF HEAD OF MESSAGES
print(df['msg'].head(5).apply(text_process))

# INSTANTIATE COUNT VECTORIZER AND FIT
bow_transformer = CV().fit(df['msg'])

# LENGTH OF VOCABULARY IN VECTORIZER
print(len(bow_transformer.vocabulary_))

# GET MESSAGE NUMBER 4 UNANLTERED
mess4 = df['msg'].iloc[3]
print(mess4)

# GET BAG OF WORDS FROM THE FITTED TRANSFORMER (VECTORIZER)
bow4 = bow_transformer.transform([mess4])
print(bow4)

print(bow4.shape)

# SEE THE WORDS WHICH WHERE IN THE MESSAGE TWICE (THESE RESULTS ARE DIFFERENT FROM THE LECTURE - I SUSPECT THAT SCIKITLEARN WAS SIMPLY UPDATED)
print(bow_transformer.get_feature_names()[6679])

# PART 3 - PROCESSING
# TRANSFORM THE ENTIRE MESSAGES COLUMN
messages_bow = bow_transformer.transform(df['msg'])

# PRINT CHARACTERISTICS
print(messages_bow.shape)
print(messages_bow.nnz)

# THE SPARSITY CALCULATION
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print(sparsity)

# TFIDF TRANSFORMATION
tt = TT().fit(messages_bow)
tfidf4 = tt.transform(bow4)

print(tfidf4)

# TFIDF TRANSFORMATION OF BOW TRANSFORMATION (THE COUNT VECTORIZED MESSAGES)
messages_tfidf = tt.transform(messages_bow)

# NAIVE BAYES
spam_detect_model = MNB().fit(messages_tfidf,df['label'])
pred4 = spam_detect_model.predict(tfidf4)[0]
print(pred4)

pred = spam_detect_model.predict(messages_tfidf)

rate = np.mean(pred == df['label'])
print("Rate: {}\n".format(rate))

# TRAIN AND TEST
msg_train,msg_test,label_train,label_test = TTS(df['msg'],df['label'],test_size=0.3,random_state=64)

# PIPELINE - A WAY TO STORE DATA PREPARATION PIPELINE
pipe = Pipeline([
	('bow',CV(analyzer=text_process)), # COUNT VECTORIZER
	('tfidf',TT()), # TFIDF TRANSFORMER
	('classifier',MNB())
])

# FIT AND PREDICT - BUSINESS AS USUAL
pipe.fit(msg_train,label_train)

pred_pipe = pipe.predict(msg_test)

rate = np.mean(pred_pipe == label_test)
print("Rate: {}\n".format(rate))

print(CR(label_test,pred_pipe))