import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from sklearn.model_selection import train_test_split as TTS
from sklearn.preprocessing import StandardScaler as SS
from sklearn.metrics import confusion_matrix as CM

plt.style.use('ggplot')

# READ THE DATA
dir = "Bootcamp/DeepLearning/"
df = pd.read_csv(dir+'bank_note_data.csv')
print(df.head())
print(df.info())
print(df.describe())

# PLOTTING
# sns.countplot(x='Class',data=df)
# plt.show()

# sns.pairplot(df)
# plt.show()

# PREPROCESSING
ss = SS().fit(df.drop('Class',axis=1)) # INSTANTIATE AND FIT SCALER
X = pd.DataFrame(ss.transform(df.drop('Class',axis=1)),columns=df.drop('Class',axis=1).columns) # TRANSFORM INTO X
y = df['Class'] # GET Y
X_train,X_test,y_train,y_test = TTS(X,y,test_size=0.3,random_state=42) # SPLIT

# TENSORFLOW
feat_cols = [tf.feature_column.numeric_column(col) for col in X_train.columns] # FEATURE COLUMNS
classifier = tf.estimator.DNNClassifier(hidden_units=[10,15,20,15,10],feature_columns=feat_cols) # INSTANTIATE THE CLASSIFIER
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,shuffle=True) # INPUT FUNCTION - IT FEEDS THE DATA TO THE ESTIMATOR OBJECT
classifier.train(input_fn=input_func,steps=500) # TRAIN THE ESTIMATOR
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,shuffle=False) # THE INPUT FUNCTION FOR FEEDING THE DATA TO THE CLASSIFIER DURING PREDICTION
preds = classifier.predict(input_fn=pred_fn) # GET PREDICTION
final_preds = [pred['class_ids'][0] for pred in preds] # GET VALUES FROM PREDICTION

print(CM(y_test,final_preds))


