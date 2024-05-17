# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:04:46 2024

@author: Yamaç
"""

#Importing the libraries
from ucimlrepo import fetch_ucirepo 

import tensorflow
import pandas
import numpy

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

#Concatenating X and y to drop y rows with Nan values
X = pandas.DataFrame(numpy.concatenate((X, y), axis=1))

#Dropping the rows with Nan values
X = X.dropna()
y = pandas.DataFrame(X.iloc[:, -1])

#Making y binary
for i in (range(len(y))):
    if y.iloc[i, 0] != 0:
        y.iloc[i, 0] = 1

X = pandas.DataFrame(X.drop(13,axis='columns'))
#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
#Normalizing the data
columns_to_scale = [0,3,4,7,9,11]

X_train[columns_to_scale] = preprocessing.normalize(X_train[columns_to_scale])
X_test[columns_to_scale] = preprocessing.normalize(X_test[columns_to_scale])

#Creating the ANN
model = tensorflow.keras.models.Sequential()

model.add(tensorflow.keras.layers.Dense(units=32, activation='relu'))

model.add(tensorflow.keras.layers.Dense(units=64, activation='relu'))

model.add(tensorflow.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=16, epochs=100)

y_pred = model.predict(X_test)

for i in range(len(y_pred)):
    if y_pred[i] >= 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
    
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

f1 = f1_score(y_test, y_pred, average='macro')  # Macro-averaging
print("F1-score (Macro):", f1)