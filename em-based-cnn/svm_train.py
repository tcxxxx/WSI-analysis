'''
	Python 3.5
'''

import pickle
import os
import csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing

from PIL import Image
from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import datasets, neighbors, linear_model
from sklearn.svm import SVC
from sklearn import datasets, neighbors, linear_model
from sklearn.multiclass import OutputCodeClassifier


'''
	to-do: histogram
'''

'''
	Classification
'''
# kNN
knn = neighbors.KNeighborsClassifier()
# Logistic Regression
logistic = linear_model.LogisticRegression()

print('KNN score: %f' % knn.fit(train_ft, train_label).score(test_ft, test_label))
print('LogisticRegression score: %f'
      % logistic.fit(train_ft, train_label).score(test_ft, test_label))

# SVM
list_of_acc=list()

accur=0
# for c in np.logspace(-2, 10, 5):
c=1000
# for c in np.logspace(-2, 10, 5):
#     for c in np.logspace(-2, 10, 5):
for c in [100, 1000, 10000, 100000]:
    for g in np.logspace(-9, 3, 13):

        clf = OutputCodeClassifier(svm.SVC(random_state=0, gamma=g, C=c),
                code_size=10, random_state=0)

        accur_temp = clf.fit(svmtrain, svmtrainlabel).score(svmtest, svmtestlabel)

        if accur < accur_temp:
            accur = accur_temp
            gamma = g

        print(c, g, accur)

list_of_acc.append(accur)
print(np.mean(list_of_acc))


