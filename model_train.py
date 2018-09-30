# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 17:32:23 2018

@author: zbj
"""
from sklearn.cross_validation import train_test_split
from stack_classifier import *
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
import matplotlib.pyplot as plt
import time
import pandas as pd


## 总样本
Y = train_byZero['tag']
X = train_byZero.iloc[:,3:5081]

## 缺失值进行填充
#X.ffillna(-999,inplace=True)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0)


## 参数
train_param = { 'max_depth': 6, 'learning_rate': 0.2, 'n_estimators': 1000, 'objective': 'binary:logistic', 'colsample_btree': 0.8 }

where_store_classifier_model= root + 'xgboost.model'

## 建立模型
xgboo = XgboostClassifier(
                          where_store_classifier_model = where_store_classifier_model,
                          train_params = train_param
                          )

start_time = time.time()
xgboo.build_model()
xgboo.fit(train_x = train_x, train_y = train_y)

time = time.time() - start_time
print('model train time:', time)
xgboo.save_model()

'''
计算特征重要性
'''

''' ## 报错了 需要解决
### plot feature importance
fig,ax = plt.subplots(figsize=(30,30))
plot_importance(xgboo,height=0.5,ax=ax,max_num_features=120) 
plt.show()
'''

## 测试模型准确率
preb_y = xgboo.predict(test_x)
accuracy = accuracy_score(test_y, preb_y)
print("accuarcy: %.2f%%" % (accuracy*100.0))

## 测试模型准确率
preb_y_pro = xgboo.predict_proba(test_x)

## 读取验证集
valid_data = pd.read_table('E:\\liuhongbing\\360finance\\open_data_train_valid1\\open_data_train_valid\\valid.txt',sep='\t')
valid_data_filter = valid_data[train_x.columns]

preb_valid_pro = xgboo.predict_proba(valid_data_filter)

valid_id = pd.read_table('E:\\liuhongbing\\360finance\\open_data_train_valid1\\open_data_train_valid\\valid_id.txt',sep='\t')

valid_id['prob'] = preb_valid_pro[:,1]
valid_id.to_csv('E:\\liuhongbing\\360finance\\open_data_train_valid1\\open_data_train_valid\\result.txt',sep=',',index=None)
