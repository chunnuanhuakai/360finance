# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:31:47 2018

@author: zbj
"""

import pandas as pd
import matplotlib.pyplot as plt


'''
train_sample = pd.read_table(root+"train_sample.txt", sep='\t')
## 降维: 当特征90%的都为0时，删除
def getColumnsByZero():    
    tmp = data_info.loc['90%']>0    
    tmp1 = tmp[tmp.values]    
    return tmp1.index

    
data_info = train_sample.describe(percentiles=[.25,.5,.90,])
colByZero = getColumnsByZero()
data_info = data_info[colByZero] 
data_info.loc['3sigma'] = data_info.loc['std']*3 + data_info.loc['mean']
data_info.to_csv(root+'data_sum_info.txt', sep='\t',index=None)
'''


root = "E:\\liuhongbing\\360finance\\open_data_train_valid1\\open_data_train_valid\\train\\"
data_info = pd.read_table(root +'data_sum_info.txt', sep='\t')
#
#data_info.drop(['Unnamed: 0'],axis=1,inplace=True)
#data_info.index = pd.Index(['count','mean','std','min','25%','50%','90%','max','3sigma'])
#data_info.reindex()



## 定义 特征的最大值 最小值
max_columns = data_info.loc['3sigma']
min_columns = data_info.loc['min']
max_min_columns = max_columns.astype(float) - min_columns.astype(float) + 0.01

## 取得需要填充的 均值
data_info_mean = data_info.loc['mean'].astype(float).to_dict()


## 读取数据的header
train_header = pd.read_table(root +'train_1.txt', sep='\t',chunksize=10)
train_header = train_header.get_chunk(5)
train_header = train_header.columns


'''
## 对样本缺失值进行填充
train_sample_1 = pd.read_table(root +'train_1.txt', sep='\t')
train_sample_1 = train_sample_1[colByZero]
train_sample_1.fillna(data_info_mean, inplace=True)
## 对样本outline 进行处理
train_sample_1.loc['3sigma'] = data_info.loc['3sigma']
train_sample_1.drop('3sigma', axis=0, inplace=True)


## 对样本2 缺失值进行填充
train_sample_2 = pd.read_table(root +'train_2.txt', sep='\t',header=None)
train_sample_2.columns = train_header

train_sample_2 = train_sample_2[colByZero]
train_sample_2.fillna(data_info_mean, inplace=True)


## 对样本outline 进行处理[3sigma 原则]
train_sample_2_outline = train_sample_2.iloc[0:20001,2:4028] - data_info.loc['3sigma']
train_sample_2_outline[train_sample_2_outline>0] = 0
train_sample_2_result = train_sample_2_outline.iloc[0:20001,:] + data_info.loc['3sigma']

## 数据归一化[观测是否是正态分布: 不符合正态分布] 利用最大值最小值进行处理。其中最大值利用 3sigma
## 查看是否正态分布
plt.hist(train_sample_2_result['f28'], bins=30, rwidth=0.9, normed=True)
plt.show()

train_sample_2_norm = train_sample_2_result.iloc[0:20001,:] - min_columns
train_sample_2_norm = train_sample_2_norm.iloc[0:20001,:]/max_min_columns

train_sample_2_norm['id'] = train_sample_2['id']
train_sample_2_norm['tag'] = train_sample_2['tag']

train_sample_2_norm.to_csv(root+'train_2_norm.txt', sep='\t',index=None)
'''

'''
data_norm: 对样本输入进行处理
1：均值填充
2：异常值处理
3：数据归一化
'''
inputpath = root+'train_1.txt'
output = root+'train_1_norm.txt'

def data_norm(inputpath, output):
    ## 对样本2 缺失值进行填充
    train_sample_2 = pd.read_table(inputpath, sep='\t',header=None,low_memory=False)
    train_sample_2.columns = train_header
    train_sample_2 = train_sample_2[data_info.columns]
    train_sample_2.fillna(data_info_mean, inplace=True)
    
    
    ## 对样本outline 进行处理[3sigma 原则]
    train_sample_2_outline = train_sample_2.iloc[0:-1,:].astype(float) - data_info.loc['3sigma']
    train_sample_2_outline[train_sample_2_outline>0] = 0
    train_sample_2_result = train_sample_2_outline.iloc[0:-1,:] + data_info.loc['3sigma']
    
    ## 数据归一化[观测是否是正态分布: 不符合正态分布] 利用最大值最小值进行处理。其中最大值利用 3sigma
    '''
    plt.hist(train_sample_2_result['f28'], bins=30, rwidth=0.9, normed=True)
    plt.show()
    '''
    train_sample_2_norm = train_sample_2_result.iloc[0:-1,:] - min_columns
    train_sample_2_norm = train_sample_2_norm.iloc[0:-1,:]/max_min_columns
    
    train_sample_2_norm['id'] = train_sample_2['id']
    train_sample_2_norm['tag'] = train_sample_2['tag']
    
    train_sample_2_norm.to_csv(output, sep='\t',index=None)
    
data_norm(root+'train_2.txt', root+'train_2_norm.txt') 
data_norm(root+'train_3.txt', root+'train_3_norm.txt')     
data_norm(root+'train_4.txt', root+'train_4_norm.txt') 
data_norm(root+'train_5.txt', root+'train_5_norm.txt') 



def valid_data_norm(inputpath, output):
    inputpath = 'E:\liuhongbing\\360finance\\open_data_train_valid1\\open_data_train_valid\\valid.txt'
    valid_data = pd.read_table(inputpath, sep='\t',low_memory=False)
    data_info_valid = data_info.drop(['tag'], axis=1)
    valid_data = valid_data[data_info_valid.columns]
    ## 定义 特征的最大值 最小值
    max_columns = data_info_valid.loc['3sigma']
    min_columns = data_info_valid.loc['min']
    max_min_columns = max_columns.astype(float) - min_columns.astype(float) + 0.01
    
    ## 取得需要填充的 均值
    data_info_mean = data_info_valid.loc['mean'].astype(float).to_dict()
    
    valid_data.fillna(data_info_mean, inplace=True)
    ## 对样本outline 进行处理[3sigma 原则]
    valid_data_outline = valid_data.iloc[0:-1,:].astype(float) - data_info_valid.loc['3sigma']
    valid_data_outline[valid_data_outline>0] = 0
    valid_data_result = valid_data_outline.iloc[0:-1,:] + data_info_valid.loc['3sigma']

    valid_data_norm = valid_data_result.iloc[0:-1,:] - min_columns
    valid_data_norm = valid_data_norm.iloc[0:-1,:]/max_min_columns
    
    valid_data_norm['id'] = valid_data['id']
    
    output = 'E:\liuhongbing\\360finance\\open_data_train_valid1\\open_data_train_valid\\valid_norm.txt'
    valid_data_norm.to_csv(output, sep='\t',index=None)
    
'''
1:去掉空值占比 80%的特征
2：去掉0值占比  90%的特征
3： 数据归一化
   [1-去掉异常数据  2-空值利用均值填充  3-数据归一化]
4：低方差滤波： 方差变化比较小，特征影响不明显
'''
        
def train_sample_merge():
    
    train_1 = pd.read_table(root+'train_1_norm.txt', sep='\t')
    train_2 = pd.read_table(root+'train_2_norm.txt', sep='\t')
    train_3 = pd.read_table(root+'train_3_norm.txt', sep='\t')
    train_4 = pd.read_table(root+'train_4_norm.txt', sep='\t')
    train_5 = pd.read_table(root+'train_5_norm.txt', sep='\t')
    train_sum = pd.concat([train_1, train_2, train_3, train_4, train_5])
    
    train_info = train_sum.describe()
    train_info.to_csv(root+'train_sum_info.txt', sep='\t')

    train_sum.to_csv(root+'train_sum.txt', sep='\t',index=None)
    # 可视化方差变化率
    '''
    plt.hist(train_info.loc['std'][2:-1], bins=30, rwidth=0.9, normed=True)
    plt.show()
    '''
    ## 去掉方差变化率在 0.02以下的特征
    train_info_std = train_info.loc['std']
    train_info_std = train_info_std[train_info_std>0.02]

    train_sum_filter_02 = train_sum[train_info_std.index]    
    train_sum_filter_02.to_csv(root+'train_sum_filter_02.txt', sep='\t',index=None)
    
    ## 去掉方差变化率在 0.05以下的特征
    train_info_std = train_info.loc['std']
    train_info_std = train_info_std[train_info_std>0.05]

    train_sum_filter_05 = train_sum[train_info_std.index]    
    train_sum_filter_05.to_csv(root+'train_sum_filter_05.txt', sep='\t',index=None)
















