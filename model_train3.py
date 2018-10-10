# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:21:02 2018

@author: zbj
"""

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score


## 总样本
root = "E:\\liuhongbing\\360finance\\open_data_train_valid1\\open_data_train_valid\\train\\";
train_by_zero = pd.read_table(root+'train_sample.txt', sep='\t')
Y = train_by_zero['tag']
X = train_by_zero.iloc[:,3:5081]
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0)

#
#tuned_parameters= [{'n_estimators':[100,200,500],
#                  'max_depth':[3,5,7], ##range(3,10,2)
#                  'learning_rate':[0.5, 1.0],
#                  'subsample':[0.75,0.8,0.85,0.9]
#                  }]
tuned_parameters= [{'n_estimators':[100,200,500,1000]
                  }]
                  
train_param = {'max_depth':6,       # 树深度
               'learning_rate':0.3, # 学习率
               'nthread':4,
               'min_child_weight':1, #叶子节点最小权重
               'gamma':0.1,
               'subsample':1,       # 全部的样本进行训练
               'reg_lambda':1,      # 正则话参数
               'colsample_btree':0.8,  # 80%的特征
               'n_estimators':100,    
               'scale_pos_weight':2,   ## 正负样本比例
               'seed':100
               }
                  
clf = GridSearchCV(XGBClassifier(**train_param), 
                   param_grid=tuned_parameters,
                   scoring='roc_auc',
                   n_jobs=4,
                   iid=False,
                   cv=5)
  
clf.fit(train_x, train_y)
##clf.grid_scores_, clf.best_params_, clf.best_score_
print(clf.best_params_)

pred_y = clf.predict(test_x)
accuracy = accuracy_score(test_y, pred_y)
print("accuarcy: %.2f%%" % (accuracy*100.0))


y_proba=clf.predict_proba(test_x)[:,1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(test_y, y_proba))             
