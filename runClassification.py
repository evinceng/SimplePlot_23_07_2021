# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 18:06:09 2021

@author: Andrej Košir
"""

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn import svm
from sklearn.datasets import fetch_openml
import scipy.stats as scs



# @brief gets generic effect size
# @arg x 1D array of data
# @arg y labels od groups, two classes only
def get_generic_es(x, y):
    
    labels = np.unique(y)
    M = len(labels)
    if M != 2:
        return 0
    
    x_G1 = x[y==labels[0]]
    x_G2 = x[y==labels[1]]
    
    n1, n2 = len(x_G1), len(x_G2)
    mu1, mu2 = np.mean(x_G1), np.mean(x_G2)
    sd1, sd2 = np.std(x_G1), np.std(x_G2)
    
    pooled_sd = np.sqrt(((n1-1)*sd1*sd1 + (n2-1)*sd2*sd2) / (n1 + n2 - 2))

    gen_es = np.abs(mu2-mu1) / pooled_sd
    
    return gen_es

x = np.array([1,2,3,4,5,6,5,4,3,2,1]) # data
y = np.array([2,3,3,2,2,3,2,3,2,2,3]) # labels
p_sd = get_generic_es(x, y)
print (p_sd)




# @brief compute P, R, F class by class for a given conf mat
# @arg CM: confusion matrix
# 
def get_PRF_from_CM(CM):
    
    return 1


# @brief return expected confusion matrix of a random classifier
# @arg cls_sizes_lst a list of classification class sizes [n1, n_2, ...]
def get_rand_cls_CM(cls_sizes_lst):
    M = len(cls_sizes_lst)
    CM = np.zeros((M, M))
    for ii in range(M):
        CM[ii, :] = (cls_sizes_lst[ii]/M)*np.ones(M)
    return CM
#cls_sizes_lst=  [12, 10, 20]
#CM = get_rand_cls_CM(cls_sizes_lst)
#print (CM)

# @brief get p-values per class of a given confusion matrix with H0=[classification is random]
def get_pvals_percls_from_CM(CM):
    
    M = CM.shape[0]
    rand_CM = get_rand_cls_CM(CM.sum(axis=1))
    p_vals = np.zeros(M) 
    
    for ii in range(M):
        
        # Perform Chi-Square Goodness of Fit Test
        observed = CM[ii, :]
        expected = rand_CM[ii, :]
        s, p = scs.chisquare(f_obs=observed, f_exp=expected)
        p_vals[ii] = p
        
        # UseZ-test - might be stronger
        
    return p_vals
#CM = np.array([[20, 11, 6], [8, 19, 3], [11, 2, 21]])
#p_vals = get_pvals_percls_from_CM(CM)
#print (p_vals)


#%% Cross validation 



# Load data
X, y = datasets.load_iris(return_X_y=True)
#X, y = fetch_openml(data_id=1464, return_X_y=True)



clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=10)


scores_F1 = cross_val_score(
    clf, X, y, cv=10, scoring='f1_macro')

print ('F1: ', np.mean(scores_F1))



#%% Confusion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold



# Simple version - not correct
y_pred = cross_val_predict(clf, X, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)


# Correct folding
M = len(np.unique(y))
all_CM = np.zeros((M,M))

kf = KFold(n_splits=10, shuffle=True, random_state=0)
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    
    # Get folds
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Get this fold confusion matrix 
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    curr_CM = confusion_matrix(y_test, y_predict, labels=[0, 1, 2])
    
    # Sum confusion matrices
    if curr_CM.shape[0] == M:
        all_CM += curr_CM
    
print (all_CM)


#%% Get characteristics 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_score = clf.decision_function(X_train)
R = recall_score(y_test, y_pred, average='micro')
P = precision_score(y_test, y_pred, average='micro')
F1 = f1_score(y_test, y_pred, average='micro')
print ('(P, R, F1) = ', (P, R, F1))



#%% PR curve
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
#y_score = clf.decision_function(X_train)
#prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])
#pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()


#%% ROC curve
from sklearn.metrics import roc_curve, plot_roc_curve
from sklearn.metrics import RocCurveDisplay

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
clf = svm.SVC(random_state=0)
clf.fit(X_train, y_train)

#plot_roc_curve(clf, X_test, y_test, pos_label=clf.classes_[1])  

#y_score = clf.decision_function(X)
#fpr, tpr, _ = roc_curve(y, y_score, pos_label=clf.classes_[1])
#roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()





   
   
#%% Significance of confusion matrix

# Goodnes of fit significant difference 
p_vals = get_pvals_percls_from_CM(all_CM)
print (p_vals)


