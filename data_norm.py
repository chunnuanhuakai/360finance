# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 09:47:35 2018

@author: zbj
"""


import pandas as pd
import numpy as np

root= "E:\\liuhongbing\\360finance\\open_data_train_valid1\\open_data_train_valid\\train\\";

train_1 = pd.read_table(root+'train_1.txt', sep='\t')
train_2 = pd.read_table(root+'train_2.txt', sep='\t',header=None)
train_3 = pd.read_table(root+'train_3.txt', sep='\t',header=None)
train_4 = pd.read_table(root+'train_4.txt', sep='\t',header=None)
train_5 = pd.read_table(root+'train_5.txt', sep='\t',header=None)

train_sum = pd.concat([train_2, train_3, train_4, train_5])
train_sum.columns = train_1.columns
train_sum = pd.concat([train_sum, train_1])


data_info_bak = train_sum.describe(percentiles=[.25,.5,.95,.98])
data_info_bak.loc['count_un_null'] = data_info_bak.loc['count']/100000

## 去掉 NULL占比98%的特征
def getColumnsByUnNull():    
    tmp = (data_info_bak.loc['count_un_null']>0.2)    
    tmp1 = tmp[tmp.values]    
    return tmp1.index   
    
colByUnNull = getColumnsByUnNull()
train_byUnNull = train_sum[colByUnNull]
data_info_byUnNull = data_info_bak[colByUnNull]


'''
colByUnNullList = colByUnNull.tolist()

colByUnNull = pd.DataFrame({'colu':colByUnNull, 'value':np.range(len(colByUnNull))})
colByUnNull = pd.DataFrame({'colu':colByUnNull, 'value':np.arange(len(colByUnNull))})
colByUnNull.to_csv(root+"train\\colByUnNull.txt", sep='\t',index=None)
'''


## 去掉0 占比 98%的特征
def getColumnsByZero():    
    tmp = data_info_byUnNull.loc['98%']>0    
    tmp1 = tmp[tmp.values]    
    return tmp1.index
    
    
colByZero = getColumnsByZero()
train_byZero = train_byUnNull[colByZero]

## 把 特征记录下来
colByZero = pd.DataFrame({'col':colByZero, 'value':np.arange(len(colByZero))})
colByZero.to_csv(root+"colByZero_0929.txt", sep='\t',index=None)

train_byZero.head()


## 含tag 的样本
train_byZero_tag_0 = train_byZero[train_byZero['tag']==0]
train_byZero_tag_1 = train_byZero[train_byZero['tag']==1]

train_byZero.to_csv(root+'train_sample.txt', sep='\t',index=None)


'''
## 计算 含逾期未逾期的样本
## 含标签样本
train_sample = train_byZero[~np.isnan(train_byZero.label)]
train_sample_positive = train_sample[train_sample.label==0]
train_sample_negative = train_sample[train_sample.label==1]

print('正负样本比例：', len(train_sample_positive.index)/len(train_sample_negative.index))
rateposneg = len(train_sample_positive.index)/len(train_sample_negative.index)
print('正负样本比例：', rateposneg)
train_sample_tag_0 = train_sample[train_sample.tag==0]
train_sample_t0_posi = train_sample_tag_0[train_sample_tag_0.label==0]
train_sample_t0_nega = train_sample_tag_0[train_sample_tag_0.label==1]
print('通过用户正负样本：', len(train_sample_t0_posi)/len(train_sample_t0_nega))
train_sample_tag_1 = train_sample[train_sample.tag==1]
train_sample_t1_posi = train_sample_tag_1[train_sample_tag_1.label==0]
train_sample_t1_nega = train_sample_tag_1[train_sample_tag_1.label==1]
print('未通过用户正负样本：', len(train_sample_t1_posi)/len(train_sample_t1_nega))

len(train_sample_positive)/len(train_sample)

len(train_sample_tag_0)
len(train_sample_tag_1)
train_byZero.to_csv(root+"train_sample.txt", sep='\t',index=None)
train_sample_bytag = train_byZero[~np.isnan(train_byZero.tag)]
train_sample_bytag_posi = train_sample_bytag[train_sample_bytag.tag==0]
train_sample_bytag_nega = train_sample_bytag[train_sample_bytag.tag==1]
train_sample_bytag_posi.to_csv(root+'train_sample_bytag_posi', sep='\t',index=None)
'''