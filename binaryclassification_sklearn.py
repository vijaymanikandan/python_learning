#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 13:01:12 2018

@author: vjanakir
"""


from sklearn.datasets import load_breast_cancer

filename = '/data2/vijay/procedureWork/modelMissRate/trial_data.csv'

import pandas as pd, numpy as np
#read_iterator = pd.read_csv(filename, usecols=colidx, skiprows=1, chunksize=chunk_size)
#count = 0
#for chunk in read_iterator:
#        data = chunk.values


# load data
fields = ['Window_Size', 'Magnitude_of_Excursion(ft)']
df = pd.read_csv(filename, usecols = fields)
header = list(df.columns)
data = np.array(df.values, dtype = float)
x = data[:,0]
y = np.zeros(x.shape)
y[data[:,1] > 300] = 1
y[data[:,1] < -300] = -1

selidx = np.where(~np.isnan(x))[0]
x = x[selidx]
y = y[selidx]

selidx = np.where(y>=0)[0]
x = x[selidx]
y = y[selidx]

x = np.expand_dims(x, -1)
y = np.expand_dims(y, -1)

# analysis
np.where(y==0)[0].shape[0]*1./y.shape[0]
np.where(y==1)[0].shape[0]*1./y.shape[0]

# data centering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Split our data
from sklearn.model_selection import StratifiedShuffleSplit as split
str_split = list(split(n_splits=1, test_size=0.3, random_state=1).split(x, y))[0]
trainidx, testidx = str_split[0], str_split[1]

xtrain = x[trainidx,:]
ytrain = y[trainidx,:]
xtest = x[testidx,:]
ytest = y[testidx,:]


seed = 1
cv_flag = 1
cv_budget = 500
kfold = 10

# cross validation
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

# linear model simple:
lr_model = lr(penalty='l1', C= 0.01, class_weight='balanced', max_iter=1000)
lr_model.fit(xtrain, ytrain)

ypred = lr_model.predict(xtest)

metrics.classification_report(ytest,ypred)
metrics.confusion_matrix(ytest, ypred)
metrics.roc_auc_score(ytest, ypred)
metrics.precision_score(ytest, ypred)

if cv_flag == 1:
    maxCVsize = min(100000, xtrain.shape[0])
#    minval = min(1000, maxCVsize/20)
#    print(xtrain.shape)
#    param_grid = {
#          "min_samples_split": np.arange(1000,5000,500).tolist(),
#          "max_depth": np.arange(1,21).tolist(),
#          "min_samples_leaf": np.arange(minval,minval*10,minval).tolist(),
#          "max_leaf_nodes": np.arange(5,26).tolist(),
#          "min_impurity_decrease": [0.01, 0.001, 0.0001]
#          }
#    dtc_model = dtc(criterion='gini', class_weight='balanced', random_state=seed)
    
    model = lr(class_weight='balanced')
    param_grid = {
            "penalty": ['l1', 'l2'],
            "C": np.logspace(-4,4,9,base = 10.0),
            "max_iter": np.arange(50,10000,50),
            }
    
    random_search = RandomizedSearchCV(model, scoring='roc_auc', param_distributions=param_grid, n_iter=cv_budget, n_jobs=-1, verbose=0, cv=kfold)
    random_search.fit(xtrain[:min(xtrain.shape[0],maxCVsize*kfold),:], ytrain[:min(xtrain.shape[0],maxCVsize*kfold),0])
    
    # Utility function to report best scores
    def report(results, n_top=1):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
    report(random_search.cv_results_)
    
    best_idx=np.where((random_search.cv_results_['rank_test_score']==1)==True)[0][0]
    testScoreopt=random_search.cv_results_['mean_test_score'][best_idx]
    
#    min_samples_split=random_search.cv_results_['params'][best_idx]['min_samples_split']
#    max_depth=random_search.cv_results_['params'][best_idx]['max_depth']
#    min_samples_leaf=random_search.cv_results_['params'][best_idx]['min_samples_leaf']
#    max_leaf_nodes=random_search.cv_results_['params'][best_idx]['max_leaf_nodes']
#    min_impurity_decrease=random_search.cv_results_['params'][best_idx]['min_impurity_decrease']
    
    C = random_search.cv_results_['params'][best_idx]['C']
    penalty = random_search.cv_results_['params'][best_idx]['penalty']
    maxiter = random_search.cv_results_['params'][best_idx]['max_iter']
    
    
else:
    model = dtc(criterion='gini', max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=0.01, random_state=seed, min_samples_leaf=minleaf, class_weight='balanced', min_weight_fraction_leaf=0., min_samples_split=1000) # max_leaf_nodes=8
    model.fit(xtrain[:min(xtrain.shape[0],maxCVsize*kfold),:],ytrain[:min(xtrain.shape[0],maxCVsize*kfold),:])

# best model retrain
#model = dtc(criterion='gini',
#                   class_weight='balanced',
#                   random_state=seed,
#                   min_samples_split=min_samples_split,
#                   max_depth=max_depth,
#                   min_samples_leaf=min_samples_leaf,
#                   max_leaf_nodes=max_leaf_nodes,
#                   min_impurity_decrease=min_impurity_decrease,
#                   )

model = lr(class_weight='balanced', C = C, max_iter=maxiter, penalty=penalty)
model.fit(xtrain,ytrain)

ypred = model.predict(xtest)

# Evaluate accuracy
metrics.classification_report(ytest,ypred)
metrics.confusion_matrix(ytest, ypred)
metrics.roc_auc_score(ytest, ypred)
metrics.precision_score(ytest, ypred)