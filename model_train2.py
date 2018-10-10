from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

## 总样本
root = "E:\\liuhongbing\\360finance\\open_data_train_valid1\\open_data_train_valid\\train\\";
train_by_zero = pd.read_table(root+'train_sample.txt', sep='\t')
Y = train_by_zero['tag']
X = train_by_zero.iloc[:,3:5081]
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0)

## 建立xgb model
train_param = {'max_depth':6,
               'learning_rate':0.3, 
               'nthread':4,
               'min_child_weight':1,
               'gamma':0.1,
               'subsample':1,
               'reg_lambda':1,
#               'reg_alpha':0,
               'n_estimators':150,
               'scale_pos_weight':2,
               'seed':100
               }
clf = XGBClassifier(**train_param)

clf.fit(train_x,train_y)
pred_y = clf.predict(test_x)


accuracy = accuracy_score(test_y, pred_y)
print("accuarcy: %.2f%%" % (accuracy*100.0))


## 模型报存
joblib.dump(clf, root + 'xgboost.model_100')
## 加载模型
clf2 = joblib.load(root + 'xgboost.model_100')

## 准确率
pred_y2 = clf2.predict(test_x)
accuracy = accuracy_score(test_y, pred_y2)
print("accuarcy: %.2f%%" % (accuracy*100.0))
print('150_accuracy : 85.88%')
print('200_accuracy: 85.62%')
print('100_accuracy: 85.24%')