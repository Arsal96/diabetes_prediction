'''
****    DONT REMOVE THIS SECTION     *****

 This is ML code developed for diabetes prediction.
 Author: Arslan
 Developing Date: 11-Nov-2018, 10 PM
 
 NOTE: I merely created a wrapper for better and simplicity. Original code were developed by SusanLi


'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


file = pd.read_csv('diabetes.csv')

x_train, x_test, y_train, y_test = train_test_split(file.loc[:, file.columns != 'Outcome'],
                                                    file['Outcome'], stratify=file['Outcome'], random_state=66)


dT = DecisionTreeClassifier(max_depth=3, random_state=0)
dT.fit(x_train, y_train)

print("Training Accuracy: {:.3f}".format(dT.score(x_train, y_train)))
print("Training Accuracy: {:.3f}".format(dT.score(x_test, y_test)))

# Accuracy of Decision Tree on training set: 0.773
# Accuracy of Decision Tree on test set: 0.740