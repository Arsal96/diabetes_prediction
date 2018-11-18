'''
****    DONT REMOVE THIS SECTION     *****

 This is ML code developed for diabetes prediction.
 Author: Arslan
 Developing Date: 11-Nov-2018, 10 PM
 
 NOTE: I merely created a wrapper for better and simplicity. Original code were developed by SusanLi

'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


file = pd.read_csv('diabetes.csv')

x_train, x_test, y_train, y_test = train_test_split(file.loc[:, file.columns != 'Outcome'],
                                                    file['Outcome'], stratify=file['Outcome'], random_state=66)

svc = SVC()
svc.fit(x_train, y_train)

print("training accuracy {:.3f}" .format(svc.score(x_train, y_train)))
print("Testing Score {:.3f}" .format(svc.score(x_test, y_test)))


# Accuracy of SVM classifier on training set: 1.000
# Accuracy of SVM classifier on test set: 0.651
