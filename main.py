import time
import numpy as np
import scipy.io as sio
import pandas as pd
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from preprocess import ### sth here

def KNN(train_X, train_y, k):
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(train_X, train_y)
    
    return clf

def SVM(train_X, train_y):
    clf = SVC(kernel = 'rbf', random_state = 0, gamma = 1, C = 1) ### need change values here
    clf.fit(train_X, train_y)

    return clf

def Rand_Forest(train_X, train_y):
    clf = RandomForestClassifier()
    clf.fit(train_X, train_y)

    return clf

if __name__ == '__main__':

    # loading from preprocess.py

    train_X = 
    train_y = # ground truth
    test_X = 
    test_y = # ground truth

    # sklearn.metrics.precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)[source]
    # sklearn.metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)[source]

    # ---------------- KNN ---------------- #
    start_time = time.time()
    knn_classifier = KNN(train_X, train_y, k = 10) # create KNN classifier ### need change k here
    end_time = time.time()

    y_pred = knn_classifier.predict(test_X)

    print("Training Time: %s seconds" % (end_time - start_time))

    print("Accuracy of KNN: " , accuracy_score(test_y, y_pred, normalize=False))
    print("Precision of KNN: ", precision_score(test_y, y_pred), average='weighted')
    print("Recall of KNN: ", recall_score(test_y, y_pred, average='weighted'))


    # ---------------- SVM ---------------- #
    start_time = time.time()
    svm_classifier = SVM(train_X, train_y) # create SVM classifier
    end_time = time.time()

    y_pred = svm_classifier.predict(test_X)

    print("Training Time: %s seconds" % (end_time - start_time))

    print("Accuracy of SVM: " , accuracy_score(test_y, y_pred, normalize=False))
    print("Precision of SVM: ", precision_score(test_y, y_pred, average='weighted'))
    print("Recall of SVM: ", recall_score(test_y, y_pred, average='weighted'))

    # ----------- Random Forest ----------- #
    start_time = time.time()
    rfc_classifier = Rand_Forest(train_X, train_y) # create random forest classifier
    end_time = time.time()

    y_pred = rfc_classifier.predict(test_X)

    print("Training Time: %s seconds" % (end_time - start_time))
    
    print("Accuracy of Rand Forest: ", accuracy_score(test_y, y_pred, normalize=False))
    print("Precision of Rand Forest: ", precision_score(test_y, y_pred, average='weighted'))
    print("Recall of Rand Forest: ", recall_score(test_y, y_pred, average='weighted'))
