
# coding: utf-8

# In[ ]:


import time
import preprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from numpy import log, random
import numpy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier


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

#Compute the error terms of each classifier
def compute_error(train_X, train_y):
    clf1 = KNeighborsClassifier(n_neighbors = 10)
    clf1.fit(train_X, train_y)
    error1 = 1 - accuracy_score(train_y, clf1.predict(train_X))
    
    clf2 = SVC(kernel = 'rbf', random_state = 0, gamma = 1, C = 1) ### need change values here
    clf2.fit(train_X, train_y)
    error2 = 1 - accuracy_score(train_y, clf2.predict(train_X))

    clf3 = RandomForestClassifier()
    clf3.fit(train_X, train_y)
    error3 = 1 - accuracy_score(train_y, clf3.predict(train_X))
    
    return [error1,error2,error3]

#Boosting Method
def adaBoost(train_X, train_y):
    #Split the data into 10 parts
    X_split = []
    y_split = []
    n = 0
    for i in range(0, len(train_X), int(len(train_X)/10)):
        #print(X_split)
        X_split.append( train_X[i:min(i+int(len(train_X)/10), len(train_X))] )
        y_split.append( train_y[i:min(i+int(len(train_X)/10), len(train_X))] )
        n = n + 1

    #initialize the weight of each tuple in W to 1/d, i.e. the probability of choosing a particular set
    W = [1/10]*10   #i.e. D = [1/10, 1/10, ...]
    
    W_clf = [0,0,0]
    
    for i in range(0,20): #for each round
        pt = int(random.choice(10, 1, p=W, replace=True))
        error = list()
        error = compute_error(X_split[pt], y_split[pt])
        mean_error = sum(error)/3 
        if mean_error > 0.5:
            continue
        
        ##Update the weight
        W[pt] = W[pt]*mean_error/(1-mean_error)
        sum_W = sum(W)
        
        ##Normalize the weight within [0,1]
        W = W/sum_W
            

        
        ##for each classifier, update their weights
        for k in range (0,3):
            W_clf[k] += log((1-error[k])/error[k])
            
    print(W_clf)
    return W_clf


if __name__ == '__main__':

    # loading from preprocess.py

    train_X , train_y = preprocess.main(True)
    test_X , test_y = preprocess.main(False)

    # sklearn.metrics.precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)[source]
    # sklearn.metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)[source]

    # ---------------- KNN ---------------- #
    
    start_time = time.time()
    knn_classifier = KNN(train_X, train_y, k = 20) # create KNN classifier ### need change k here
    end_time = time.time()

    y_train_pred = knn_classifier.predict(train_X)
    y_test_pred = knn_classifier.predict(test_X)
    
    
    print("Training Time: %s seconds" % (end_time - start_time))

    print("Accuracy of KNN for training: " , accuracy_score(train_y, y_train_pred))
    print("Accuracy of KNN for testing: " , accuracy_score(test_y, y_test_pred))
    #print("Precision of KNN for training: ", precision_score(train_y, y_train_pred), average='weighted')
    #print("Precision of KNN for testing: ", precision_score(test_y, y_test_pred), average='weighted')
    #print("Recall of KNN for training: ", recall_score(train_y,  y_train_pred, average='weighted'))
    #print("Recall of KNN for testing: ", recall_score(test_y, y_test_pred, average='weighted'))


    # ---------------- SVM ---------------- #
    start_time = time.time()
    svm_classifier = SVM(train_X, train_y) # create SVM classifier
    end_time = time.time()

    y_train_pred = svm_classifier.predict(train_X)
    y_test_pred = svm_classifier.predict(test_X)

    print("Training Time: %s seconds" % (end_time - start_time))

    print("Accuracy of SVM for training: " , accuracy_score(train_y, y_train_pred))
    print("Accuracy of SVM for testing: " , accuracy_score(test_y, y_test_pred))
    print("Precision of SVM for training: ", precision_score(train_y, y_train_pred, average='weighted'))
    print("Precision of SVM for testing: ", precision_score(test_y, y_test_pred, average='weighted'))
    print("Recall of SVM for training: ", recall_score(train_y, y_train_pred, average='weighted'))    
    print("Recall of SVM for testing: ", recall_score(test_y, y_test_pred, average='weighted')) 

    # ----------- Random Forest ----------- #
    start_time = time.time()
    rfc_classifier = Rand_Forest(train_X, train_y) # create random forest classifier
    end_time = time.time()

    y_training_pred = rfc_classifier.predict(train_X)
    y_test_pred = rfc_classifier.predict(test_X)

    print("Training Time: %s seconds" % (end_time - start_time))
    
    print("Accuracy of Rand Forest for training: ", accuracy_score(train_y, y_training_pred))
    print("Accuracy of Rand Forest for testing: ", accuracy_score(test_y, y_test_pred))
    print("Precision of Rand Forest for training: ", precision_score(train_y, y_training_pred, average='weighted'))
    print("Precision of Rand Forest for testing: ", precision_score(test_y, y_test_pred, average='weighted'))
    print("Recall of Rand Forest for training: ", recall_score(train_y, y_training_pred, average='weighted'))
    print("Recall of Rand Forest for testing: ", recall_score(test_y, y_test_pred, average='weighted'))
    
    
    # -----------Boosting ----------- #
    W_ens = adaBoost(train_X, train_y)
    #return the weights represnting KNN, SVM, Random Forest, respectively
    sum_W_ens = sum(W_ens) 
    for i in range(0,len(W_ens)):
        W_ens[i] /= sum_W_ens
    print(W_ens)
    
    clf1 = KNN(train_X, train_y, 20)
    clf2 = SVM(train_X, train_y)
    clf3 = Rand_Forest(train_X, train_y)

    y_test_pred = (clf1.predict(test_X)).astype(int)*W_ens[0] + (clf2.predict(test_X)).astype(int)*W_ens[1] + (clf3.predict(test_X)).astype(int)*W_ens[2]
    y_test_pred2 = around(y_test_pred).astype(int).astype(str)
    y_test_pred2.tolist()
    
    print("Accuracy of Ensemble for testing: ", accuracy_score(test_y, y_test_pred2))
    print("Precision of Ensemble for testing: ", precision_score(test_y, y_test_pred2, average='weighted'))
    print("Recall of Ensemble for testing: ", recall_score(test_y, y_test_pred2, average='weighted'))


# In[ ]:


#Cross Validation Techniques
k_range = range(1,20)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, train_X, train_y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

print (k_scores)


## plot for choosing k
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.plot(k_range, k_scores)
plt.axis("equal")
plt.ylabel('Cross validation Score')

plt.show()

