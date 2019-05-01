'''
Project: Predicting Loan Status
Script: Random Forest Classification
Auther: Yanxiang Ding
Date: April 2019

Data used: All Lending Club loan data
    (https://www.kaggle.com/wordsforthewise/lending-club)
'''


# ========== import packages ==========
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn import metrics
import time
import matplotlib.pyplot as plt
%matplotlib qt5


# ========== import data ==========
folder_path = path
test_x = pd.read_csv(folder_path+'test_x.csv', sep='\t', index_col=0)
test_y = pd.read_csv(folder_path+'test_y.csv', sep='\t', index_col=0)
test_y['level_cor'] = test_y.apply(lambda row: 1 if row['level_cor']==1 else
                                   (2 if row['level_cor']==2 else 
                                    (3 if row['level_cor']==5 else 4)),axis=1)
test_y = test_y['level_cor']

train_x_ori = pd.read_csv(folder_path+'train_x.csv', sep='\t', index_col=0)
train_y_ori = pd.read_csv(folder_path+'train_y.csv', sep='\t', index_col=0)
train_y_ori['level_cor'] = train_y_ori.apply(lambda row: 1 if row['level_cor']==1 else
                                             (2 if row['level_cor']==2 else
                                              (3 if row['level_cor']==5 else 4)),axis=1)
train_y_ori = train_y_ori['level_cor']


# ========== model selection (parameter tuning ==========
X = np.array(train_x_ori)
y = np.array(train_y_ori)

n_estimators_ = [10,50,100,200,400]
max_depth_ = [5,10,20,30,50]
min_leaf_ = [5,10,15,25,40,70]
train_precision = []
train_recall = []
train_f1 = []
test_precision = []
test_recall = []
test_f1 = []
for estimator in n_estimators_:
    result = [estimator]
    RF = RandomForestClassifier(min_samples_leaf=estimator,class_weight='balanced',max_depth=15,n_estimators=150,random_state=123)
    RF.fit(train_x_ori, train_y_ori)
    train_measure = precision_recall_fscore_support(train_y_ori, RF.predict(train_x_ori))
    test_measure = precision_recall_fscore_support(test_y,RF.predict(test_x))
    train_precision.append(list(train_measure[0]))
    train_recall.append(list(train_measure[1]))
    train_f1.append(list(train_measure[2]))
    test_precision.append(list(test_measure[0]))
    test_recall.append(list(test_measure[1]))
    test_f1.append(list(test_measure[2]))
    print('{}/{} done'.format(n_estimators_.index(estimator)+1,len(n_estimators_)))


# ========== visualise results ==========
test_precision = pd.DataFrame(test_precision,columns=['1','2','3','4'])
test_recall = pd.DataFrame(test_recall,columns=['1','2','3','4'])
test_f1 = pd.DataFrame(test_f1,columns=['1','2','3','4'])

fig = plt.figure()
ax1 = fig.add_subplot(131)
for i in range(1,5):
    ax1.plot(n_estimators_,test_precision[str(i)])
    ax1.set_xlabel('min samples leaf')
    ax1.set_title('Precision')
ax2 = fig.add_subplot(132)
for i in range(1,5):
    ax2.plot(n_estimators_,test_recall[str(i)])
    ax2.set_xlabel('min samples leaf')
    ax2.set_title('Recall')
ax3 = fig.add_subplot(133)
for i in range(1,5):
    ax3.plot(n_estimators_,test_f1[str(i)])
    ax3.set_xlabel('min samples leaf')
    ax3.set_title('F1 score')
    ax3.legend(['Charged-Off','Fully Paid','Late 31-120 Days','Late 0-30 Days'])
fig.suptitle('Random Forest Parameter Selection - min samples leaf',fontsize=18)


# ========== compute train time of best model ==========
RF = RandomForestClassifier(min_samples_leaf=15,class_weight='balanced',max_depth=15,n_estimators=150,random_state=123)

start = time.time()
RF.fit(train_x_ori, train_y_ori)
end = time.time()
print(end - start)
test_measure = precision_recall_fscore_support(test_y,RF.predict(test_x))
print(test_measure[2])
    
