import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import re
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import xgboost
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['A','C','G','T','Z']))
#Z for variables that are not A C G T , 
#Variables other than 4 characters are regarded equally.

#========== label encoder to float numbers
def ordinal_encoder(my_array):
    integer_encoded = label_encoder.transform(my_array)
    float_encoded = integer_encoded.astype(float)
    float_encoded[float_encoded == 0] = 0.25 # A
    float_encoded[float_encoded == 1] = 0.50 # C
    float_encoded[float_encoded == 2] = 0.75 # G
    float_encoded[float_encoded == 3] = 1.00 # T
    float_encoded[float_encoded == 4] = 0.00 # anything else, z
    return float_encoded

#=========== change the string sequence to array
def string_to_array(my_string):
    my_string = re.sub('[^ACGT]', 'Z', my_string)
    my_array = np.array(list(my_string))
    return my_array


#===== Read Data with Pandas
data = pd.read_csv("names.data",names = ["label", "attributes", "sequence"])
data2 = data.copy()

#===== Change String Sequence into Array
data2["sequence"]= data2["sequence"].apply(string_to_array)
print("Applying String to Array")

#===== Applying Ordinal encoding on Array[ sequence ]

data2["sequence"]= data2["sequence"].apply(ordinal_encoder) 
print("Applying Ordinal encoding")
data2["sequence"]= tf.keras.preprocessing.sequence.pad_sequences(np.array(data2.sequence), maxlen=None, padding='post', dtype='float32',value=0.0).tolist()

#===== Converts sequence column to a series
labeldata= data2.sequence.apply(pd.Series)
labeldata["label"] = data2["label"]

#======= gets rid of the label column to prevent cheating#
X = labeldata.drop(['label'], axis = 1)
X_trial = labeldata.drop(['label'], axis = 1).loc[0]
y = labeldata['label']
print(X_trial)

# Splits of training data/testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)

#Callbacks
from keras.callbacks import EarlyStopping
monitor = tf.keras.callbacks.EarlyStopping(monitor ='val_loss' , mode ='auto', patience=20, verbose=0 )
callbacks=[monitor]

#Converts Data for custom predicting
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def create_model():
    # create model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(120,input_dim=X.shape[1], activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#Invoking  KerasClassifier
estimator = KerasClassifier(build_fn=create_model, epochs = 200, batch_size= 20, verbose=0, callbacks=callbacks, validation_split=0.20) 
estimator.fit(X_train, y_train)
#Predicting
y_pred = estimator.predict(sc.transform(np.array([X_trial])), verbose=1)
print(y_pred)
