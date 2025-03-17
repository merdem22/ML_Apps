import numpy as np
import pandas as pd
import math


X_train = np.genfromtxt("20newsgroup_words_train.csv", delimiter = ",", dtype = int)
y_train = np.genfromtxt("20newsgroup_labels_train.csv", delimiter = ",", dtype = int)
X_test = np.genfromtxt("20newsgroup_words_test.csv", delimiter = ",", dtype = int)
y_test = np.genfromtxt("20newsgroup_labels_test.csv", delimiter = ",", dtype = int)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    #goal: class_priors[i] = Pr(y=i+1)
    #the number of classes is 20, so initailize with that size 

    K = 20
    N = len(y)

    class_priors = np.zeros(K)

    for label in y:
        class_priors[label - 1] +=1

    class_priors/=N #divide each label sum with N to get class mean.
    
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
#print(class_priors)



# STEP 4
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_success_probabilities(X, y):
    # your implementation starts below

    #set the env variables.
    a = 0.2
    K = 20
    D = 2000

    counts = np.zeros((K, D))
    class_counts = np.zeros(K)

    for x_vector, label in zip(X, y): #itates over each data point
        label_index = label - 1
        class_counts[label_index] +=1
        counts[label_index] += x_vector #apparently we can do vector addition in numpy, so we don't have to enter a for loop of size D.

    P = np.empty((K,D))
    for i in range(K):
        P[i] = (counts[i] + a) / (class_counts[i] + a*D)    

    # your implementation ends above
    return(P)

P = estimate_success_probabilities(X_train, y_train)
#print(P)



# STEP 5
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, P, class_priors):
    # your implementation starts below
    
    K = 20
    D = 2000
    N = len(X)

    #precompute these for efficiency.
    score_values = np.zeros((N, K))
    log_estimate = np.log(P)
    inverse_log_estimate = np.log(1 - P)
    log_class_priors = np.log(class_priors)

    #using for loops takes forever, usning matmul from the library is better
    #also, turned multiplication into a sum to avoid float underflow.
    for i, x_vector in enumerate(X):
        score_values[i] += np.matmul(log_estimate, x_vector) + np.matmul(inverse_log_estimate, 1 - x_vector) + log_class_priors
    
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, P, class_priors)
#print(scores_train)

scores_test = calculate_score_values(X_test, P, class_priors)
#print(scores_test)



# STEP 6
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    K = 20
    confusion_matrix = np.zeros((K, K))
    
    for i, score in enumerate(scores):
        #get predicted label
        prediction = np.argmax(score)
        confusion_matrix[prediction, y_truth[i] - 1] += 1

    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print("Training accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_train)) / np.sum(confusion_train)))

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print("Test accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_test)) / np.sum(confusion_test)))
