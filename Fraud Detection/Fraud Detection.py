# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 09:52:43 2021

@author: Karthik
"""

import os

os.chdir("H:\Data Science\Assignments\Fraud Detection")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


#from google.colab import files
#uploaded = files.upload()
df = pd.read_csv('payment_fraud.csv')
df.head()


# Split dataset up into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('label', axis=1), df['label'],
    test_size=0.33, random_state=17)


clf = LogisticRegression().fit(X_train, y_train)

# Make predictions on test set
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))



