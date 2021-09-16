#!/usr/bin/env python
# coding: utf-8

# In[1]:


import openpyxl
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pickle

from sklearn import preprocessing
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable, gradcheck

print("setup successfully!")


# In[3]:


#8月10日更新  读均值

MIN_LEN = 96   #第121用户只有96个数据
USERNUM = 208

t = 1
d = 3

train_data = []

#时延嵌入提取用户的均值
temp_features = []
for i in range(USERNUM):
    fpath = 'base/'+str(i+1)+'.csv'
    user = pd.read_csv(fpath, header = None, usecols = [0,1,2,3,4,5])
    
    #print(np.where(np.isnan(user)))
    #第一列feature 时延嵌入
    for k in range(MIN_LEN-d+1):
        temp_f = user.iloc[k:k+d,0].values.tolist()
        temp_features.append(temp_f)
    temp_avg_feature = np.array(temp_features)
    temp_features=[]
    user_avg_feature = np.expand_dims(temp_avg_feature, axis=1)
    #后五列feature 时延嵌入 
    for j in range(5):
        for k in range(MIN_LEN-d+1):
            temp_f = user.iloc[k:k+d,j+1].values.tolist()
            temp_features.append(temp_f)
        temp_avg_feature = np.array(temp_features)
        temp_features=[]
        user_avg_feature = np.concatenate((user_avg_feature,np.expand_dims(temp_avg_feature, axis=1)),axis=1)
   # print("user_avg_feature.shape:{}".format(user_avg_feature.shape))
    temp_user_avg_feature = np.expand_dims(user_avg_feature, axis=0)
   #print("temp_user_avg_feature.shape:{}".format(temp_user_avg_feature.shape))
    if i == 0:
        users_avg_feature = temp_user_avg_feature
    else:
        users_avg_feature = np.concatenate((users_avg_feature,temp_user_avg_feature),axis=0)
print("users_avg_feature.shape:{}".format(users_avg_feature.shape))

#时延嵌入提取用户右腿的均值
temp_features = []
for i in range(USERNUM):
    fpath = 'base/'+str(i+1)+'.csv'
    user = pd.read_csv(fpath, sep=',', header = None, usecols = [12,13,14,15,16,17])
    #第一列feature 时延嵌入
    for k in range(MIN_LEN-d+1):
        temp_f = user.iloc[k:k+d,0].values.tolist()
        temp_features.append(temp_f)
    temp_right_feature = np.array(temp_features)
    temp_features=[]
    user_right_feature = np.expand_dims(temp_right_feature, axis=1)
    #后五列feature 时延嵌入 
    for j in range(5):
        for k in range(MIN_LEN-d+1):
            temp_f = user.iloc[k:k+d,j+1].values.tolist()
            temp_features.append(temp_f)
        temp_right_feature = np.array(temp_features)
        temp_features=[]
        user_right_feature = np.concatenate((user_right_feature,np.expand_dims(temp_right_feature, axis=1)),axis=1)
    #print("user_right_feature.shape:{}".format(user_right_feature.shape))
    temp_user_right_feature = np.expand_dims(user_right_feature, axis=0)
    #print("temp_user_right_feature.shape:{}".format(temp_user_right_feature.shape))
    if i == 0:
        users_right_feature = temp_user_right_feature
    else:
        users_right_feature = np.concatenate((users_right_feature,temp_user_right_feature),axis=0)
print("users_right_feature.shape:{}".format(users_right_feature.shape))


#提取正常腿均值
normal_user_feature = []
fpath = 'data/normal.csv'
#fpath = 'normal_minus_base.csv'
user = pd.read_csv(fpath, sep=',', header = None, skiprows=[0], names =['fan', 'xuan', 'qu', 'qian', 'shang', 'nei'])
user = user.values
normal_user_feature.append(user[:MIN_LEN])
 
    
#提取用户的类别信息
fpath = 'data/label.csv'
user_label = pd.read_csv(fpath, sep=',', header = None, names =['y'])
user_label = user_label.values
#将原来的4分类，转为2类，即只有当左腿不好，才认为有问题
for index,label in enumerate(user_label):
    if label ==2:
        user_label[index] = 0
    if label == 3:
        user_label[index] = 1
print("Finish Load Data!")


# In[4]:



import scipy.stats as stats

users_feature = np.array(users_avg_feature)-np.array(users_right_feature)
#print("users_feature.shape:{}".format(users_feature.shape))

users_feature2 = np.array(users_avg_feature)
users_feature3 = np.array(users_right_feature)

#print(np.array(users_mean_feature)[0,:,0].shape)

user_feature = np.concatenate((np.array(users_feature), np.array(users_feature2), np.array(users_feature3)), axis=2)
#将以上两类特征，合并，每个用户的每条数据为18维


from sklearn import preprocessing

#对用户数据的每一列进行归一化，即按照['fan', 'xuan', 'qu', 'qian', 'shang', 'nei']，分别进行处理
for i in range(user_feature.shape[3]):
    for k in range(user_feature.shape[2]):
        for j in range(user_feature.shape[0]):
            if j == 7 or j == 45 or j == 162:
                continue
                temp = user_feature[j,:,k,i].reshape(1, -1)
                temp = preprocessing.normalize(temp, norm ='l2')  #对特征值进行归一化处理 
                #temp = preprocessing.scale(temp)
                print(temp.shape)
                users_feature[j,:,k,i] = temp.reshape(-1)
        
user_feature = np.delete(user_feature, (7,45,162), 0)  #直接将这些用户剔除
user_label = np.delete(user_label, (7,45,162,208), 0)  #同时剔除无效的标签

org_users_label = user_label
org_users_feature = user_feature


print(user_feature.shape)


# In[5]:


print(user_label.shape)


# In[49]:


path = 'data/pkl/org_users_label.pkl'
pickle.dump(org_users_label,open(path,'wb+'))


# In[50]:


path = 'data/pkl/org_users_feature.pkl'
pickle.dump(user_feature,open(path,'wb+'))

