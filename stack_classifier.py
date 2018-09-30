# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 16:17:29 2018

@author: zbj
"""
from sklearn.externals import joblib
import pickle
# 定义分类器
class Classifier(object):
    
    '''
    定义分类器接口
    '''
    def __init__(self, where_store_classifier_model=None, train_params=None):
        """
        :param where_store_classifier_model:模型保存路径
        :param train_params: 训练参数
        """
        self.classifier_model_path = where_store_classifier_model
        self.train_params = {} if train_params is None else train_params
        
        
    def fit(self, train_x, train_y):
        """
        拟合数据
        :return:
        """
        raise RuntimeError("need to implement!")
    
    def predict(self, test_x):
        """
        预测标签
        :param test_x:
        :return:
        """
        raise RuntimeError("need to implement!")
        
        
    def predict_proba(self, test_x):
        """
        预测标签概率(分布)
        :param test_x:
        :return:
        """
        raise RuntimeError("need to implement!")
        
    def save_model(self):
        """
        存储模型
        :return:
        """
        raise RuntimeError("need to implement!")

    def load_model(self):
        """
        加载模型
        :return:
        """
        raise RuntimeError("need to implement!")
        
        
class SklearnClassifier(Classifier):
    """
    基于sklearn api的classifier实现
    """
    def __init__(self, where_store_classifier_model=None, train_params=None, classifier_class=None):
        Classifier.__init__(self, where_store_classifier_model, train_params)
        self.classifier_class = classifier_class

    def build_model(self):
        self.classifier_model = self.classifier_class(**self.train_params)

    def fit(self, train_x, train_y):
        self.class_num = len(set(train_y))
        self.classifier_model.fit(train_x, train_y)

    def predict(self, test_x):
        return self.classifier_model.predict(test_x)
        
    def predict_proba(self, test_x):
        return self.classifier_model.predict_proba(test_x)
        
    def save_model(self):
        joblib.dump(self.classifier_model, self.classifier_model_path)
        with open(self.classifier_model_path + '_class_num', 'wb') as w_class_num_file:
            pickle.dump(self.class_num, w_class_num_file)
            
    def load_model(self):
        self.classifier_model = joblib.load(self.classifier_model_path)
        with open(self.classifier_model_path + '_class_num', 'rb') as r_class_num_file:
            self.class_num = pickle.load(r_class_num_file)


            
class RandomForestClassifier(SklearnClassifier):
    def __init__(self, where_store_classifier_model=None, train_params=None):
        from sklearn.ensemble import RandomForestClassifier
        SklearnClassifier.__init__(self, where_store_classifier_model, train_params, RandomForestClassifier)


class GradientBoostingClassifier(SklearnClassifier):
    def __init__(self, where_store_classifier_model=None, train_params=None):
        from sklearn.ensemble import GradientBoostingClassifier
        SklearnClassifier.__init__(self, where_store_classifier_model, train_params, GradientBoostingClassifier)


class AdaBoostClassifier(SklearnClassifier):
    def __init__(self, where_store_classifier_model=None, train_params=None):
        from sklearn.ensemble import AdaBoostClassifier
        SklearnClassifier.__init__(self, where_store_classifier_model, train_params, AdaBoostClassifier)
    
        
class XgboostClassifier(SklearnClassifier):
    def __init__(self, where_store_classifier_model=None, train_params=None):
        from xgboost import XGBClassifier
        SklearnClassifier.__init__(self, where_store_classifier_model, train_params, XGBClassifier)
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        

        