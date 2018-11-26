import time
import preprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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

    train_X , train_y = preprocess.main(True)
    test_X , test_y = preprocess.main(False)

    # sklearn.metrics.precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)[source]
    # sklearn.metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)[source]

    # ---------------- KNN ---------------- #
    start_time = time.time()
    knn_classifier = KNN(train_X, train_y, k = 10) # create KNN classifier ### need change k here
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

