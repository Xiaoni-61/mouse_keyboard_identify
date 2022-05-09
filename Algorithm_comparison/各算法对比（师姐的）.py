#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
# import sys
# reload(sys)
# sys.setdefaultencoding('gb18030')
# path='E:\libsvm-3.23'
# sys.path.append(path)
import pandas as pd
from sklearn import svm, tree, ensemble, naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data0 = pd.read_excel("2_1_newdata.xls", header=None)
data1 = pd.read_csv("2_newkeydata_1.txt", header=None, sep=",", encoding="ISO-8859-1")
from sklearn.decomposition import PCA, LatentDirichletAllocation

key = data1.iloc[0:, 1:]

data = pd.concat([data0, key], axis=1, ignore_index=True)
'''
label=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1]
user=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
'''
label = [-1, 1, -1, -1]
user = [1, 2, 3, 4]
data[0].replace(user, label, inplace=True)
data2 = data[(data[0] == -1)]
data3 = data[(data[0] == 1)]
data4 = data2.sample(n=data3.shape[0], axis=0, random_state=60)
data5 = pd.concat([data3, data4], axis=0, ignore_index=True)
data5 = data5.dropna(axis=0, how='any')
Y1 = data5.iloc[0:, [0]].values
Y1 = Y1.ravel()
X2 = data5.iloc[0:, 1:].values

X = MinMaxScaler()

X1 = X.fit_transform(X2)

# train/test split
Xtr, Xte, Ytr, Yte = train_test_split(X2, Y1, test_size=.15, random_state=60)
# Xtr2,Xte2,Ytr2,Yte2 = train_test_split(X2,Y2, test_size=.30, random_state=42)


Ytr = Ytr.astype('int')
Yte = Yte.astype('int')

'''
##SVM
grid=svm.SVC(cache_size=5000,kernel='linear')
grid.fit(Xtr,Ytr)
pred=grid.predict(Xte)
precision = precision_score(Yte, pred)
print ('mouse precision score: %.3f' % (precision))
recal=recall_score(Yte,pred)
print ('mouse recall score: %.3f' % (recal))
f1=f1_score(Yte,pred)
print ('mouse f1 score: %.3f' % (f1))
accuracy=accuracy_score(Yte,pred)
print ('mouse accuracy score: %.3f' % (accuracy))
report=classification_report(Yte,pred)
print (report)
cfm=confusion_matrix(Yte,pred)
print (cfm)
cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
print (cfm)
print ('done')
'''
'''
#params={'priors':[0.5,0.5]}
gnb=naive_bayes.GaussianNB(priors=None).fit(Xtr,Ytr)
score=gnb.score(Xte,Yte)
pred=gnb.predict(Xte)
print ('Testscore: %.2f' % (score))
precision = precision_score(Yte, pred)
print ('mouse precision score: %.3f' % (precision))
recal=recall_score(Yte,pred)
print ('mouse recall score: %.3f' % (recal))
f1=f1_score(Yte,pred)
print ('mouse f1 score: %.3f' % (f1))
accuracy=accuracy_score(Yte,pred)
print ('mouse accuracy score: %.3f' % (accuracy))
report=classification_report(Yte,pred)
print (report)
cfm=confusion_matrix(Yte,pred)
print (cfm)
cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
print (cfm)
print ('done')
'''

'''
##决策树
params={'max_depth':range(1,21),'min_samples_split':range(2,20),'criterion':np.array(['entropy','gini'])}
#grid=GridSearchCV(tree.DecisionTreeClassifier(random_state=10),param_grid=params,cv=5,scoring='roc_auc')
#grid=GridSearchCV(tree.DecisionTreeClassifier(random_state=10),param_grid=params,cv=5,scoring='f1')
grid=GridSearchCV(tree.DecisionTreeClassifier(random_state=10),param_grid=params,cv=5)
grid.fit(Xtr,Ytr)
pred=grid.predict(Xte)
print ('bestparameters: %s, bestscore: %.2f,Testscore: %.2f' % (grid.best_params_,grid.best_score_,grid.score(Xte,Yte)))
precision = precision_score(Yte, pred)
print ('mouse precision score: %.3f' % (precision))
recal=recall_score(Yte,pred)
print ('mouse recall score: %.3f' % (recal))
f1=f1_score(Yte,pred)
print ('mouse f1 score: %.3f' % (f1))
accuracy=accuracy_score(Yte,pred)
print ('mouse accuracy score: %.3f' % (accuracy))
report=classification_report(Yte,pred)
print (report)
cfm=confusion_matrix(Yte,pred)
print (cfm)
cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
print (cfm)
print ('done')
'''
'''
##随机森林

params={'n_estimators':range(5,50,5),'max_depth':range(1,21),'min_samples_split':range(2,20),'criterion':np.array(['entropy','gini'])}
grid=GridSearchCV(ensemble.RandomForestClassifier(random_state=10),param_grid=params,cv=5,scoring='f1')
grid.fit(Xtr,Ytr)
pred=grid.predict(Xte)
print ('bestparameters: %s, bestscore: %.2f,Testscore: %.2f' % (grid.best_params_,grid.best_score_,grid.score(Xte,Yte)))
precision = precision_score(Yte, pred)
print ('mouse precision score: %.3f' % (precision))
recal=recall_score(Yte,pred)
print ('mouse recall score: %.3f' % (recal))
f1=f1_score(Yte,pred)
print ('mouse f1 score: %.3f' % (f1))
accuracy=accuracy_score(Yte,pred)
print ('mouse accuracy score: %.3f' % (accuracy))
report=classification_report(Yte,pred)
print (report)
cfm=confusion_matrix(Yte,pred)
print (cfm)
cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
print (cfm)
print ('done')
'''

''' '''
# MKL algorithms
from MKLpy.algorithms import AverageMKL, EasyMKL, KOMD
from MKLpy.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel

# 鼠标多尺度

Xtr_mouse = Xtr[0:, 0:50]
Xte_mouse = Xte[0:, 0:50]
Xtr_keyboard = Xtr[0:, 50:]
Xte_keyboard = Xte[0:, 50:]

KLtr_mouse = []
KLte_mouse = []

KLtr_mouse.append(linear_kernel(Xtr_keyboard))
KLte_mouse.append(linear_kernel(Xte_keyboard, Xtr_keyboard))
KLtr_mouse.append(linear_kernel(Xtr_mouse))
KLte_mouse.append(linear_kernel(Xte_mouse, Xtr_mouse))
base_learner = svm.SVC(cache_size=2000)  # simil hard-margin svm
clf1 = AverageMKL(estimator=base_learner).fit(KLtr_mouse, Ytr)  # combining kernels with the EasyMKL algorithm
y_pred1 = clf1.predict(KLte_mouse)  # predictions
y_score = clf1.decision_function(KLte_mouse)  # rank
accuracy = accuracy_score(Yte, y_pred1)
roc_auc = roc_auc_score(Yte, y_score)
recal = recall_score(Yte, y_pred1)
f1 = f1_score(Yte, y_pred1)
print('easymkl mouse Accuracy score: %.3f, roc AUC score: %.3f,recall score: %.3f,f1: %.3f' % (
accuracy, roc_auc, recal, f1))
report = classification_report(Yte, y_pred1)
print(report)
cfm = confusion_matrix(Yte, y_pred1)
print(cfm)
cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
print(cfm)
print('done')

gamma = [0.001, 0.01, 0.005, 0.05, 0.1, 0.2, 0.5, 1, 5, 10, 15, 20]
C = [0.001, 0.01, 0.05, 0.1, 1, 5, 0.01, 10, 15, 70, 20, 50, 80, 100, 200, 300]
for i in gamma:
    for j in C:
        KLtr_mouse = []
        KLte_mouse = []

        KLtr_mouse.append(linear_kernel(Xtr_keyboard))
        KLte_mouse.append(linear_kernel(Xte_keyboard, Xtr_keyboard))
        KLtr_mouse.append(rbf_kernel(Xtr_mouse, gamma=i))
        KLte_mouse.append(rbf_kernel(Xte_mouse, Xtr_mouse, gamma=i))
        base_learner = svm.SVC(C=j, cache_size=2000)  # simil hard-margin svm
        clf1 = AverageMKL(estimator=base_learner).fit(KLtr_mouse, Ytr)  # combining kernels with the EasyMKL algorithm
        print(i, j)
        y_pred1 = clf1.predict(KLte_mouse)  # predictions
        y_score = clf1.decision_function(KLte_mouse)  # rank
        accuracy = accuracy_score(Yte, y_pred1)
        roc_auc = roc_auc_score(Yte, y_score)
        recal = recall_score(Yte, y_pred1)
        f1 = f1_score(Yte, y_pred1)
        print('easymkl mouse Accuracy score: %.3f, roc AUC score: %.3f,recall score: %.3f,f1: %.3f' % (
        accuracy, roc_auc, recal, f1))
        report = classification_report(Yte, y_pred1)
        print(report)
        cfm = confusion_matrix(Yte, y_pred1)
        print(cfm)
        cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
        print(cfm)
        print('done')

'''

print ('done')
best_results = {}
i=0
for lam in [0,0.1,0.4,0.8, 0.7,0.5,0.3,0.2, 0.6,0.9, 1]:
	print(i+1)
	scores = cross_val_score(KLtr_mouse, Ytr, EasyMKL(lam=lam), n_folds=5, score='roc_auc')
	acc = np.mean(scores)
	if not best_results or best_results['score'] < acc:
		best_results = {'lam' : lam, 'score' : acc}
clf = EasyMKL(lam=best_results['lam']).fit(KLtr_mouse,Ytr)		#combining kernels with the EasyMKL algorithm
print ('mouse best lam and weight:')
print(best_results['lam'])
print(best_results['score'])
print (clf.weights)
y_pred = clf.predict(KLte_mouse)					#predictions
y_score = clf.decision_function(KLte_mouse)		#rank
accuracy = accuracy_score(Yte, y_pred)
roc_auc = roc_auc_score(Yte, y_score)
recal=recall_score(Yte,y_pred)
f1=f1_score(Yte,y_pred)
print ('easymkl mouse Accuracy score: %.3f, roc AUC score: %.3f,recall score: %.3f,f1: %.3f' % (accuracy, roc_auc,recal,f1))
report=classification_report(Yte,y_pred)
print (report)
cfm=confusion_matrix(Yte,y_pred)
print (cfm)
cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
print (cfm)
print ('done')

'''

'''
scores01 = cross_val_score(KLtr_mouse, Ytr, AverageMKL(), n_folds=5, score='roc_auc')
acc01 = np.mean(scores01)
scores02 = cross_val_predict(KLtr_mouse, Ytr, AverageMKL(), n_folds=5, score='accuracy')
acc02 = np.mean(scores02)
 print ('AverageMKL  best acc:')
  print(acc01,acc02)
'''
X = MinMaxScaler()

X1 = X.fit_transform(X2)

# train/test split
Xtr, Xte, Ytr, Yte = train_test_split(X2, Y1, test_size=.15, random_state=18)
# Xtr2,Xte2,Ytr2,Yte2 = train_test_split(X2,Y2, test_size=.30, random_state=42)


Ytr = Ytr.astype('int')
Yte = Yte.astype('int')

''''''
##SVM
grid = svm.SVC(cache_size=5000, kernel='linear')
grid.fit(Xtr, Ytr)
pred = grid.predict(Xte)
precision = precision_score(Yte, pred)
print('mouse precision score: %.3f' % (precision))
recal = recall_score(Yte, pred)
print('mouse recall score: %.3f' % (recal))
f1 = f1_score(Yte, pred)
print('mouse f1 score: %.3f' % (f1))
accuracy = accuracy_score(Yte, pred)
print('mouse accuracy score: %.3f' % (accuracy))
report = classification_report(Yte, pred)
print(report)
cfm = confusion_matrix(Yte, pred)
print(cfm)
cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
print(cfm)
print('done')

''''''
# params={'priors':[0.5,0.5]}
gnb = naive_bayes.GaussianNB(priors=None).fit(Xtr, Ytr)
score = gnb.score(Xte, Yte)
pred = gnb.predict(Xte)
print('Testscore: %.2f' % (score))
precision = precision_score(Yte, pred)
print('mouse precision score: %.3f' % (precision))
recal = recall_score(Yte, pred)
print('mouse recall score: %.3f' % (recal))
f1 = f1_score(Yte, pred)
print('mouse f1 score: %.3f' % (f1))
accuracy = accuracy_score(Yte, pred)
print('mouse accuracy score: %.3f' % (accuracy))
report = classification_report(Yte, pred)
print(report)
cfm = confusion_matrix(Yte, pred)
print(cfm)
cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
print(cfm)
print('done')

''''''
##决策树
params = {'max_depth': range(1, 21), 'min_samples_split': range(2, 20), 'criterion': np.array(['entropy', 'gini'])}
# grid=GridSearchCV(tree.DecisionTreeClassifier(random_state=10),param_grid=params,cv=5,scoring='roc_auc')
# grid=GridSearchCV(tree.DecisionTreeClassifier(random_state=10),param_grid=params,cv=5,scoring='f1')
grid = GridSearchCV(tree.DecisionTreeClassifier(random_state=10), param_grid=params, cv=5)
grid.fit(Xtr, Ytr)
pred = grid.predict(Xte)
print(
    'bestparameters: %s, bestscore: %.2f,Testscore: %.2f' % (grid.best_params_, grid.best_score_, grid.score(Xte, Yte)))
precision = precision_score(Yte, pred)
print('mouse precision score: %.3f' % (precision))
recal = recall_score(Yte, pred)
print('mouse recall score: %.3f' % (recal))
f1 = f1_score(Yte, pred)
print('mouse f1 score: %.3f' % (f1))
accuracy = accuracy_score(Yte, pred)
print('mouse accuracy score: %.3f' % (accuracy))
report = classification_report(Yte, pred)
print(report)
cfm = confusion_matrix(Yte, pred)
print(cfm)
cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
print(cfm)
print('done')

''''''
##随机森林

params = {'n_estimators': range(5, 50, 5), 'max_depth': range(1, 21), 'min_samples_split': range(2, 20),
          'criterion': np.array(['entropy', 'gini'])}
grid = GridSearchCV(ensemble.RandomForestClassifier(random_state=10), param_grid=params, cv=5, scoring='f1')
grid.fit(Xtr, Ytr)
pred = grid.predict(Xte)
print(
    'bestparameters: %s, bestscore: %.2f,Testscore: %.2f' % (grid.best_params_, grid.best_score_, grid.score(Xte, Yte)))
precision = precision_score(Yte, pred)
print('mouse precision score: %.3f' % (precision))
recal = recall_score(Yte, pred)
print('mouse recall score: %.3f' % (recal))
f1 = f1_score(Yte, pred)
print('mouse f1 score: %.3f' % (f1))
accuracy = accuracy_score(Yte, pred)
print('mouse accuracy score: %.3f' % (accuracy))
report = classification_report(Yte, pred)
print(report)
cfm = confusion_matrix(Yte, pred)
print(cfm)
cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
print(cfm)
print('done')

''' '''
# MKL algorithms
from MKLpy.algorithms import AverageMKL, EasyMKL, KOMD
from MKLpy.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel

# 鼠标多尺度

Xtr_mouse = Xtr[0:, 0:50]
Xte_mouse = Xte[0:, 0:50]
Xtr_keyboard = Xtr[0:, 50:]
Xte_keyboard = Xte[0:, 50:]

KLtr_mouse = []
KLte_mouse = []

KLtr_mouse.append(linear_kernel(Xtr_keyboard))
KLte_mouse.append(linear_kernel(Xte_keyboard, Xtr_keyboard))
KLtr_mouse.append(linear_kernel(Xtr_mouse))
KLte_mouse.append(linear_kernel(Xte_mouse, Xtr_mouse))
base_learner = svm.SVC(cache_size=2000)  # simil hard-margin svm
clf1 = AverageMKL(estimator=base_learner).fit(KLtr_mouse, Ytr)  # combining kernels with the EasyMKL algorithm
y_pred1 = clf1.predict(KLte_mouse)  # predictions
y_score = clf1.decision_function(KLte_mouse)  # rank
accuracy = accuracy_score(Yte, y_pred1)
roc_auc = roc_auc_score(Yte, y_score)
recal = recall_score(Yte, y_pred1)
f1 = f1_score(Yte, y_pred1)
print('easymkl mouse Accuracy score: %.3f, roc AUC score: %.3f,recall score: %.3f,f1: %.3f' % (
accuracy, roc_auc, recal, f1))
report = classification_report(Yte, y_pred1)
print(report)
cfm = confusion_matrix(Yte, y_pred1)
print(cfm)
cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
print(cfm)
print('done')

gamma = [0.001, 0.01, 0.005, 0.05, 0.1, 0.2, 0.5, 1, 5, 10, 15, 20]
C = [0.001, 0.01, 0.05, 0.1, 1, 5, 0.01, 10, 15, 70, 20, 50, 80, 100, 200, 300]
for i in gamma:
    for j in C:
        KLtr_mouse = []
        KLte_mouse = []

        KLtr_mouse.append(linear_kernel(Xtr_keyboard))
        KLte_mouse.append(linear_kernel(Xte_keyboard, Xtr_keyboard))
        KLtr_mouse.append(rbf_kernel(Xtr_mouse, gamma=i))
        KLte_mouse.append(rbf_kernel(Xte_mouse, Xtr_mouse, gamma=i))
        base_learner = svm.SVC(C=j, cache_size=2000)  # simil hard-margin svm
        clf1 = AverageMKL(estimator=base_learner).fit(KLtr_mouse, Ytr)  # combining kernels with the EasyMKL algorithm
        print(i, j)
        y_pred1 = clf1.predict(KLte_mouse)  # predictions
        y_score = clf1.decision_function(KLte_mouse)  # rank
        accuracy = accuracy_score(Yte, y_pred1)
        roc_auc = roc_auc_score(Yte, y_score)
        recal = recall_score(Yte, y_pred1)
        f1 = f1_score(Yte, y_pred1)
        print('easymkl mouse Accuracy score: %.3f, roc AUC score: %.3f,recall score: %.3f,f1: %.3f' % (
        accuracy, roc_auc, recal, f1))
        report = classification_report(Yte, y_pred1)
        print(report)
        cfm = confusion_matrix(Yte, y_pred1)
        print(cfm)
        cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
        print(cfm)
        print('done')

'''

print ('done')
best_results = {}
i=0
for lam in [0,0.1,0.4,0.8, 0.7,0.5,0.3,0.2, 0.6,0.9, 1]:
	print(i+1)
	scores = cross_val_score(KLtr_mouse, Ytr, EasyMKL(lam=lam), n_folds=5, score='roc_auc')
	acc = np.mean(scores)
	if not best_results or best_results['score'] < acc:
		best_results = {'lam' : lam, 'score' : acc}
clf = EasyMKL(lam=best_results['lam']).fit(KLtr_mouse,Ytr)		#combining kernels with the EasyMKL algorithm
print ('mouse best lam and weight:')
print(best_results['lam'])
print(best_results['score'])
print (clf.weights)
y_pred = clf.predict(KLte_mouse)					#predictions
y_score = clf.decision_function(KLte_mouse)		#rank
accuracy = accuracy_score(Yte, y_pred)
roc_auc = roc_auc_score(Yte, y_score)
recal=recall_score(Yte,y_pred)
f1=f1_score(Yte,y_pred)
print ('easymkl mouse Accuracy score: %.3f, roc AUC score: %.3f,recall score: %.3f,f1: %.3f' % (accuracy, roc_auc,recal,f1))
report=classification_report(Yte,y_pred)
print (report)
cfm=confusion_matrix(Yte,y_pred)
print (cfm)
cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
print (cfm)
print ('done')

'''

'''
scores01 = cross_val_score(KLtr_mouse, Ytr, AverageMKL(), n_folds=5, score='roc_auc')
acc01 = np.mean(scores01)
scores02 = cross_val_predict(KLtr_mouse, Ytr, AverageMKL(), n_folds=5, score='accuracy')
acc02 = np.mean(scores02)
 print ('AverageMKL  best acc:')
  print(acc01,acc02)
'''
