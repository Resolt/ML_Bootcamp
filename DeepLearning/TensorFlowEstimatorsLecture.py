import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from sklearn.model_selection import train_test_split as TTS

from sklearn.metrics import confusion_matrix as CM

dir = "Bootcamp/DeepLearning/"

# READ DATA
df = pd.read_csv(dir+'iris.csv')

# COLUMNS CAN'T HAVE SPACE OR SPECIAL CHARACTERS
df.columns = [f.split(' (')[0].replace(' ','_') for f in df.columns]

# TARGET HAS TO BE INTEGER
df['target'] = df['target'].apply(int)

# TRAIN TEST SPLIT
X_train,X_test,y_train,y_test = TTS(df.drop('target',axis=1),df['target'],test_size=0.3)

# TENSORFLOW

# CREATE LIST OF TENSORFLOW NUMERIC FEATURE COLUMNS
feat_cols = [tf.feature_column.numeric_column(col) for col in X_train.columns]
# for col in feat_cols: print(col)

# CREATE THE INPUT FUNCTION - IT'S THE ELEMENT RESPONSIBLE FOR 'FEEDING' THE TRAINING DATA TO THE CLASSIFIER
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,num_epochs=5,shuffle=True)

# THE CLASSIFIER - THE ACUTAL NEURAL NETWORK
classifier = tf.estimator.DNNClassifier(hidden_units=[16],n_classes=3,feature_columns=feat_cols)

# THE TRAINING - THE INPUT FUNCTION FEEDS THE DATA
classifier.train(input_fn=input_func)#,steps=100)

# PREDICTION
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)

preds = list(classifier.predict(input_fn=pred_fn))

final_preds = [pred['class_ids'][0] for pred in preds]

print(CM(y_test,final_preds))