# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:06:11 2021

@author: Arpan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('spam.csv')

df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
print(df.head())

# splitting the model into test and trainsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message,df.spam,random_state=0)



#converting email body into numerical data
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_count  = cv.fit_transform(X_train.values)
X_train_count.toarray()[:2]
#print(type(X_train_count))


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)

emails=X_test.values
emails_count = cv.transform(emails)
y_predict = model.predict(emails_count)
#print(model.predict(emails_count))

from sklearn.model_selection import cross_val_score
print("accuracy")
X_test_count = cv.transform(X_test)
print(model.score(X_test_count, y_test))


print("error")
from sklearn import metrics
error=metrics.mean_squared_error( y_test, y_predict)
acc = 1 -error
print("accuracy from error : ")
print(acc)
print(metrics.mean_squared_error( y_test, y_predict))
print(metrics.mean_absolute_error(y_test, y_predict))
#emails_count=cv.tansform(X_test)
#y_predict=model.predict(emails_count)
