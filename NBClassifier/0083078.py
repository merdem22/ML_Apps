import numpy as np
import pandas as pd



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
        class_priors[int(label) - 1] +=1

    class_priors/=N #divide each label sum with N to get class mean.
    
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 4
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_success_probabilities(X, y):
    # your implementation starts below
    a = 0.2
    K = 20
    D = 2000

    class_counts = np.zeros(K) #for Nc.

    #goal: P[i][j] = the probability of word j occuring in a document of class i. Pr(xj = 1 | Y = ci) ?
    # 1 <= j <= 2000,  1<= i <=20
    P = np.zeros((K, D))

    for x, y in zip(X, y): #iterate over the data points (zip allows us to reference the label for that given x.)


    

    


    # your implementation ends above
    return(P)

P = estimate_success_probabilities(X_train, y_train)
print(P)


"""
# STEP 5
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, P, class_priors):
    # your implementation starts below
    
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, P, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, P, class_priors)
print(scores_test)



# STEP 6
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print("Training accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_train)) / np.sum(confusion_train)))

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print("Test accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_test)) / np.sum(confusion_test)))
"
"""