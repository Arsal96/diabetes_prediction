'''
****    DONT REMOVE THIS SECTION     *****

 This is ML code developed for diabetes prediction.
 Author: Arslan
 Developing Date: 11-Nov-2018, 10 PM
 
 NOTE: I merely created a wrapper for better and simplicity. Original code were developed by SusanLi
 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

file = pd.read_csv('diabetes.csv')

# apply knn alogrithm

x_train, X_test, y_train, y_test = train_test_split(
    file.loc[:, file.columns != 'Outcome'], file['Outcome'], stratify=file['Outcome'], random_state=66)

training_accuracy = []
testing_accuracy = []

knn = KNeighborsClassifier(n_neighbors=9)
# fit
knn.fit(x_train, y_train)
print(
    'Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(x_train, y_train)))
print(
    'Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))



# Accuracy of K-NN classifier on training set: 0.79
# Accuracy of K-NN classifier on test set: 0.78