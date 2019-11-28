# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:56:32 2019

@author: nohaw
"""
#itertool -> implements a number of iterator building blocks
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#Classifier implementing the k-nearest neighbors vote.
from sklearn.neighbors import KNeighborsClassifier

#No labels on the ticks(axis)
from matplotlib.ticker import NullFormatter

#module contains classes to support completely configurable tick locating and formatting.
import matplotlib.ticker as ticker

#package provides several common utility functions and transformer classes
#to change raw feature vectors into a representation that is more suitable 
from sklearn import preprocessing

from sklearn import metrics

df = pd.read_csv('teleCust1000t.csv')
df.head()

#how many of each class is in our data set
df['custcat'].value_counts()

#explore your data using visualization techniques:
df.hist(column='income', bins=50)

#Lets define feature sets, X:
df.columns

#To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]


#put labels
y = df['custcat'].values
y[0:5]

#data normalization

#Data Standardization give data zero mean and unit variance,
#it is good practice, especially for algorithms such as KNN which is based on
#distance of cases:
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

#Train Test Split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


#Training
#Lets start the algorithm with k=4

k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

#we can use the model to predict the test set:
yhat = neigh.predict(X_test)
yhat[0:5]


#Evaluate accurecy
#In multilabel classification, accuracy classification score
#is a function that computes subset accuracy. 
# the set of labels predicted for a sample must exactly 
#match the corresponding set of labels in y_true.

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


#if we increase K's

k = 6
neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat6 = neigh6.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat6))


#use the training part for modeling, and calculate the accuracy of prediction
#using all samples in your test set. Repeat this process, increasing the k, 
#and see which k is the best for your model.

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

#Plot model accuracy for Different number of Neighbors
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
