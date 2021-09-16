#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import openpyxl
from tqdm import tqdm
import pandas as pd
import math
import matplotlib.pyplot as plt
import time
import pickle
from sklearn import preprocessing
from sklearn.model_selection import KFold
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable, gradcheck
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.decomposition import PCA

print("setup successfully!")


# In[22]:


########  加载数据 ########
def loadSavedData():
    global org_users_feature,org_users_label,org_4_label,org_users_weight
    org_users_label = pickle.load(open('data/pkl/org_users_label.pkl','rb'))
    org_users_feature = pickle.load(open('data/pkl/org_users_feature.pkl','rb'))
    org_users_weight = pickle.load(open('data/pkl/users_bmi_l2.pkl','rb'))
       
    print("feature:{}".format(org_users_feature.shape))
    org_4_label = pickle.load(open('data/pkl/user_4_label.pkl','rb'))
    print("Load data successfully!")


# In[3]:


########  划分 K折 ########
def get_k_fold(k, org_users_featur,seed=1):
    kf = KFold(k,shuffle = True,random_state = seed)
    k_fold_result = kf.split(org_users_feature)
    return k_fold_result


# In[4]:


########  正负样本平均 ########
def makeDataEqual(train_index,org_test_index):
    #print("hhhhh")
    global org_users_feature,org_users_label
    global train_p_index,train_n_index,val_index,test_index
    test_index = org_test_index
    test_label = org_users_label[test_index] #测试集
    
    #对训练集打乱，选取前20为验证集
    users_train_index = [i for i in range(len(train_index))] 
    t = 1000 * time.time()
    random.seed(int(t) % 2**32)
    random.shuffle(users_train_index)
    shuffle_train_index = train_index[users_train_index]#打乱的train_index
    #print("before shuffle:{}".format(train_index))
    #print("after shuffle:{}".format(shuffle_train_index))
    #print("after shuffle:{}".format(train_index)) 分片截取后，原train——index没变
    val_index = shuffle_train_index[0:20]
    train_index = shuffle_train_index[20:]
    
    test_label = org_users_label[test_index]
    val_label = org_users_label[val_index]
    train_label = org_users_label[train_index]
    
    #调整比例
    train_p_index = np.argwhere(train_label == 1).T[0]
    train_n_index = np.argwhere(train_label == 0).T[0]
    p_sum = len(train_p_index)
    diffrent = len(train_label) - 2*p_sum
    if diffrent>0:
        #随机选取补充正样本样本
        add_p_index = np.random.randint(low=0,high=p_sum,size=diffrent)
        #生成正样本
        train_p_index = np.concatenate((train_p_index,add_p_index))
    else:
        add_n_index = np.random.randint(low=0,high=len(train_n_index),size= (-diffrent))
        train_n_index = np.concatenate((train_n_index,add_n_index))


# In[5]:


###### ---- 调整正负比例 -----######

def changeRatio(positive,negative):
    print("正例:负例={}:{}".format(positive,negative))
    global train_p_index,train_n_index,val_index,test_index,train_final_index  
    train_final_index = np.concatenate((train_p_index.repeat(positive) ,train_n_index.repeat(negative)))    


# In[6]:


#---------打乱训练集----------

def randomTrainData():
    #打乱顺序
    global train_final_index 
    t = 1000 * time.time()
    random.seed(int(t) % 2**32)
     #复制原始数据集
    users_index = [i for i in range(len(train_final_index))]  #方便生成打乱的索引
    random.shuffle(users_index)
    train_final_index = train_final_index[users_index]


# In[7]:


#---------画图----------
def pict_acc(epoches,train_acc):
    x = range(0,epoches)
    y = train_acc
    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show() 
    
    
def pict_loss(epoches,train_loss):
    x = range(0,epoches)
    y = train_loss
    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show() 


# In[8]:


"""
评估函数，放在循环一轮之后，在验证集合上检测效果

"""

def evaluation(threshold):
    global org_users_feature,org_users_label,model,DEBUG,val_index,org_users_weight
    
    pred_list=[]
    model.eval()  
   
    test_users_feature = org_users_feature[val_index]
    test_users_weight=  org_users_weight[val_index]
    test_users_label = org_users_label[val_index]
    test_num = len(test_users_feature)
    correct_num = 0

 
    for _, index in enumerate(val_index):
        
        features = Variable(torch.FloatTensor(np.expand_dims(org_users_feature[index], 0)))
        weights = Variable(torch.FloatTensor(np.expand_dims(org_users_weight[index], 0)))
        label = Variable(torch.FloatTensor(org_users_label[index]))

        if opts.use_cuda:
            features = features.cuda()
            label = label.cuda()


        feature,temp = model(features,weights)
        
        pred = nn.Sigmoid()(temp)
        pred = pred.detach().numpy()
        pred_list.append(pred)
    
    accracy, precision, recall, TNR, FNR, FPR ,check = cal_rate(pred_list,test_users_label, threshold)
    
    if (precision+recall) ==0:
        F1_score =0.0
    else:
        F1_score = 2*precision*recall / (precision + recall)
    
    EER,AUC = threshold_after_cal(pred_list,test_users_label)

    if DEBUG:
        print("pred_list:", pred_list)
    return accracy,precision,recall,F1_score,EER,-AUC,check


# In[9]:


"""
评估函数，放在循环一轮之后，检测实际效果，对训练集的测试用存储的
返回在训练集上的错误率以及预测结果，方便后续计算adaboost的系数

"""

def evaluate_train_ada(threshold,indexList,model_path):
    global model,org_users_feature,org_users_label,model,DEBUG,org_users_weight
    
    pred_list=[]
    model.eval()  
    test_users_feature = org_users_feature[indexList]
    test_users_weight = org_users_weight[indexList]
    test_users_label = org_users_label[indexList]
    test_num = len(test_users_feature)
    correct_num = 0
    
    for _, index in enumerate(indexList):

        features = Variable(torch.FloatTensor(np.expand_dims(org_users_feature[index], 0)))
        weights = Variable(torch.FloatTensor(np.expand_dims(org_users_weight[index], 0)))
        label = Variable(torch.FloatTensor(org_users_label[index]))

        if opts.use_cuda:
            features = features.cuda()
            label = label.cuda()

        feature,pred = model(features,weights) 
        pred = nn.Sigmoid()(pred)

        pred = pred.detach().numpy()
        pred_list.append(pred)
    
    accracy, precision, recall, TNR, FNR, FPR ,check = cal_rate(pred_list,test_users_label, threshold)
    
    if (precision+recall) ==0:
        F1_score =0.0
    else:
        F1_score = 2*precision*recall / (precision + recall)
    
    if DEBUG:
        print("pred_list:", pred_list)
        
    return 1-accracy ,check


# In[10]:


#--------计算相关评价指标
def cal_rate(pred,label, thres):
    check = []
    all_number = len(pred)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for item in range(all_number):
        disease = pred[item]
        if disease >= thres:
            disease = 1
        else:
            disease = 0
            
        check.append(disease == label[item])
     
        if disease == 1:
            if label[item] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if label[item] == 0:
                TN += 1
            else:
                FN += 1
   
    accracy = float(TP+TN) / float(all_number)
    if TP+FP == 0:
        precision = 0
    else:
        precision = float(TP) / float(TP+FP)
    TPR = float(TP) / float(TP+FN)
    TNR = float(TN) / float(FP+TN)
    FNR = float(FN) / float(TP+FN)
    FPR = float(FP) / float(FP+TN)
    
    return accracy, precision, TPR, TNR, FNR, FPR ,check


#------计算 EER 和 AUC

def threshold_after_cal(pred,label):

    threshold_vaule = sorted(pred)
    threshold_num = len(threshold_vaule)
    accracy_array = np.zeros(threshold_num)
    precision_array = np.zeros(threshold_num)
    TPR_array = np.zeros(threshold_num)
    TNR_array = np.zeros(threshold_num)
    FNR_array = np.zeros(threshold_num)
    FPR_array = np.zeros(threshold_num)
    # calculate all the rates
    for thres in range(threshold_num):
        accracy, precision, TPR, TNR, FNR, FPR,check = cal_rate(pred,label, threshold_vaule[thres])
        accracy_array[thres] = accracy
        precision_array[thres] = precision
        TPR_array[thres] = TPR
        TNR_array[thres] = TNR
        FNR_array[thres] = FNR
        FPR_array[thres] = FPR
    AUC = np.trapz(TPR_array, FPR_array)
    threshold = np.argmin(abs(FNR_array - FPR_array))
    EER = (FNR_array[threshold]+FPR_array[threshold])/2 #平均错误概率

    return EER,AUC


# In[11]:


############## 测试集 —— 返回check      

def test(threshold, pred_list):
    global org_users_label,DEBUG,test_index
    
    test_users_label = org_users_label[test_index]
    test_num = len(test_index)
    correct_num = 0
    #     
    accracy, precision, recall, TNR, FNR, FPR,check = cal_rate(pred_list,test_users_label, threshold)
    
    if (precision+recall) ==0:
        F1_score =0.0
    else:
        F1_score = 2*precision*recall / (precision + recall)
    
    EER,AUC = threshold_after_cal(pred_list,test_users_label)

    return accracy,precision,recall,F1_score,TNR,AUC,check


# In[12]:


#在基模型上，测试——测试集

def test_base_model(threshold,model_path):
    global org_users_feature,org_users_label,DEBUG,test_index,org_users_weight
    
    pred_list =[]
    correct_list =[]
    fail_list = []   
    test_users_feature = org_users_feature[test_index]
    test_users_weight = org_users_weight[test_index]
    test_users_label = org_users_label[test_index]
    test_num = len(test_users_feature)
    correct_num = 0
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    
    for _, index in enumerate(test_index):

        features = Variable(torch.FloatTensor(np.expand_dims(org_users_feature[index], 0)))
        weights = Variable(torch.FloatTensor(np.expand_dims(org_users_weight[index], 0)))
        label = Variable(torch.FloatTensor(org_users_label[index]))

        if opts.use_cuda:
            features = features.cuda()
            label = label.cuda()
            
        feature,pred = model(features,weights) 
        pred = nn.Sigmoid()(pred)
        pred = pred.detach().numpy()
        pred_list.append(pred)    
    
    return pred_list


# In[13]:


#————————基模型加权投票
def weighted_vote(text_index, models, my_alpha,threshold):
    ''' return final_predict based on weighted_vote of all the learners in models
        weight is the the accuracy of each learner
    '''
    n_learners = len(models)
    n_tests = len(text_index)
    probs = np.zeros(n_tests).reshape(-1,1)
    
    for i in range(n_learners):
        alpha = my_alpha[i]
        model_path = models[i]
        pred = test_base_model(threshold,model_path)  
        probs = probs + alpha * (np.array(pred).reshape(-1, 1))
  
    return probs / sum(my_alpha)


# In[14]:


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self,feat_dim, num_classes=2, use_gpu=False):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) +                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()   #####>>> z=torch.arange(2) >>> tensor([ 0,  1])
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


# In[15]:


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


# In[16]:


##### ------ 使用PCA将特征转二维，可视化
def pict_pca():
    global model,org_users_label,org_users_feature,org_users_weight,model,p_prototype
    
    model.eval()
    pca = PCA(n_components=2) 
    x1 = Variable(torch.FloatTensor(org_users_feature))
    x2 = Variable(torch.FloatTensor(org_users_weight))
    feature,_ = model(x1,x2)
    xx = pca.fit_transform(feature.detach().numpy())
    print(pca.explained_variance_ratio_)
    
    positive = np.where(org_users_label==1)[0]
    p_user = org_users_feature[positive]
    p_weight = org_users_weight[positive]
    p_user = Variable(torch.FloatTensor(p_user))
    p_weight = Variable(torch.FloatTensor(p_weight))
    feature,_ = model(p_user,p_weight)
    p_x = pca.transform(feature.detach().numpy())
    
    x = p_x[:,0]
    y = p_x[:,1]
    
    negative = np.where(org_users_label==0)[0]
    n_user = org_users_feature[negative]
    n_weight = org_users_weight[negative]
    n_user = Variable(torch.FloatTensor(n_user))
    n_weight = Variable(torch.FloatTensor(n_weight))
    feature,_ =model(n_user,n_weight)
    n_x = pca.transform(feature.detach().numpy())
    a = n_x[:,0]
    b = n_x[:,1]
    
    
    plt.scatter(x, y,marker='o')
    plt.scatter(a, b, marker='^')
    plt.show()


# In[17]:


############## ---- LSTM模块 ---- #######

######  lstm+liner outsize:20

class BaseLSTM(nn.Module):
    def __init__(self, opts):
        super(BaseLSTM, self).__init__()
        
        self.input_size = opts.feature_dim ####18
        self.pool_size = opts.lstm_pool_size
        self.output_size = opts.lstm_feature_size
        self.hidden_size = opts.lstm_hidden_size
        self.num_layer =  opts.lstm_num_layer
        
        self.layer1 = nn.LSTM(self.input_size,self.hidden_size,self.num_layer)
        self.layer2 = nn.Linear(376,opts.lstm_feature_size)
    

        #self.center = CenterLoss()
        #self.triplet = TripletLoss()

    def forward(self, input1):

        x = Variable(input1)  # torch.Size([20，94,18,3])                 
        #x_lstm = torch.mean(x, dim=3) # torch.Size([20，94,18]) 
        x_lstm,_ = self.layer1(x)
        #print("x_lstm:{}".format(x_lstm.shape))
        x_lstm = nn.MaxPool1d(self.pool_size, stride=self.pool_size)(x_lstm)
        s,b,h = x_lstm.size()
        x_lstm = x_lstm.view(s,b*h)
        x_lstm = self.layer2(x_lstm) ####### 

        return x_lstm 
    


# In[18]:



#CNN 全连接层 outsize : 220
"""
最基本的CNN网络
可以学习18维特征间的关联特征
但是对于整体96序列长的全局关系的，个人觉得捕获还不够强

"""


class BaseCNN(nn.Module):
    def __init__(self, opts):
        super(BaseCNN, self).__init__()

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)

        self.channel_num = opts.channel_num
        self.series_size = opts.series_size
        self.feature_dim = opts.feature_dim

        #self.relu = nn.LeakyReLU()
        self.stride = opts.stride
        self.embed_dropout = opts.embed_dropout
        self.fc_dropout = opts.fc_dropout
        self.len = opts.series_size
        self.out_size = opts.cnn_output_size

        self.conv = nn.Conv1d(self.len, self.channel_num, (3,1), stride=self.stride) 
        self.conv2 = nn.Conv1d(self.channel_num, self.channel_num, (3, 1), stride=self.stride)
        self.conv3 = nn.Conv1d(1, self.channel_num, (6, 3), stride=self.stride)
        self.conv4 = nn.Conv1d(self.channel_num, self.channel_num//2 , (6, 5), stride=self.stride)
        self.conv5 = nn.Conv1d(self.channel_num//2, 20, 3, 2)       
        
        in_fea = 220
       
        self.fc_dropout = nn.Dropout(self.fc_dropout)

    def forward(self, input1):

        x = Variable(input1)  # torch.Size([20，94,18,3])   
        
        
        """
        PART——1:
        
        conv、conv1 重点学习了18个变量间的关联特征，延迟嵌入的三个相邻时间维度上没有卷积
        avg_pool2d 则强化了延迟嵌入的作用，对相邻三个时间点的的特征取了平均值，这一步实现了局部特征的提取，
        同时对特征维度进行池化，减少了后续的计算量，这里使用平均池化，是因为此时处于初步特征挖掘阶段，平均池化能保留更多的信息
        
        """
        temp = self.conv(x)   # torch.Size([20，94，18, 3]) ==>  torch.Size([20，100，16,3]
        temp = self.conv2(temp)   # torch.Size([20，100，16,3]) ==>  torch.Size([20，100，14,3])
        #temp = nn.BatchNorm2d(100)(temp)
        
        out = F.relu(temp)  
        temp = F.avg_pool2d(out, kernel_size=(2,3))   
        
        """
        PART——2:
        
        temp.squeeze(3).unsqueeze(1)完成了维度的交换，使之前的学习到的100个通道的特征，转入一个通道里
        方便后续conv3、conv4对第一步学习到的关系，进行更深层次的捕获
        最后利用squeeze降维，方便后续可以使用一维卷积，减少后续卷积的参数量
        这一部分使用最大池化，是因为深挖阶段，我们着重注意最显著的特征即可
        
        """
        temp = self.conv3(temp.squeeze(3).unsqueeze(1))  # torch.Size([20,1,100，7]) ==>  torch.Size([20，100,95,5])
        temp = self.conv4(temp)   # torch.Size([20,100,95,5) ==>  torch.Size([20，50,90，1])   
        #temp = nn.BatchNorm2d(50)(temp)
        temp = F.relu(temp)  # torch.Size([20, 50, 90]) 
        
        temp = F.max_pool2d(temp, kernel_size=(2,1))  # torch.Size([20, 50, 45]) 
        temp = temp.squeeze(3) 
        """
        PART——3:
        
        利用卷积池化，对降维度，减少模型参数量，相比于最早的linear（4500，1）
        这一步的将特征维度减少到220
        
        """
        
        temp = self.conv5(temp)  # torch.Size([20, 20, 22]) 
        temp = F.relu(temp)  # torch.Size([20, 20,22])
        temp = nn.MaxPool1d(2, stride=2)(temp) # torch.Size([20, 20,11])
        out_cnn = self.fc_dropout(temp)
        out = out_cnn.reshape(out_cnn.shape[0],-1)
    
        return out 


# In[19]:


"""
卷积核大小3，步数3
与之前的差异主要在Part1，本cnn模型，注重的是对某一变量在左、右、左-右之间关系的捕获


"""

class BaseCNN2(nn.Module):
    def __init__(self, opts):
        super(BaseCNN2, self).__init__()

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)

        self.channel_num = opts.channel_num
        self.series_size = opts.series_size
        self.feature_dim = opts.feature_dim

        #self.relu = nn.LeakyReLU()
        self.stride = opts.stride
        self.embed_dropout = opts.embed_dropout
        self.fc_dropout = opts.fc_dropout
        self.len = opts.series_size
        self.out_size = opts.cnn_output_size
        
        self.conv = nn.Conv1d(self.len, self.channel_num, (3,1), stride=(3,1)) 
        self.conv2 = nn.Conv1d(self.channel_num, self.channel_num, (3, 1), stride=self.stride)
        self.conv3 = nn.Conv1d(1, self.channel_num, 6, stride=self.stride)
        self.conv4 = nn.Conv1d(self.channel_num, self.channel_num//2 , 6, stride=self.stride)
        self.conv5 = nn.Conv1d(self.channel_num//2, 20, 3, 2)       
        
        in_fea = 220
       
        self.fc_dropout = nn.Dropout(self.fc_dropout)

    def forward(self, input1):

        x = Variable(input1)  # torch.Size([20，94,18,3])                 
        #x2 = Variable(input2)
        
        temp = self.conv(x)   # torch.Size([20，94，18, 3]) ==>  torch.Size([20，100，6,3]
        temp = self.conv2(temp)   # torch.Size([20，100，6,3]) ==>  torch.Size([20，100，4,3])
        #temp = nn.BatchNorm2d(100)(temp)
        out = F.relu(temp)  
        temp = F.avg_pool2d(out, kernel_size=(4,3))   ## #####  20,100,1,1 
       
        temp = self.conv3(temp.reshape(-1,100).unsqueeze(1))  # torch.Size([20,1,100]) ==>  torch.Size([20，100,95])
        temp = self.conv4(temp)   # torch.Size([20,100,95) ==>  torch.Size([20，50,90])
        #temp = nn.BatchNorm1d(50)(temp)  
        temp = F.relu(temp)  # torch.Size([20, 50, 90]) 
        temp = nn.MaxPool1d(2, stride=2)(temp) # torch.Size([20, 50, 45]) 
        
        temp = self.conv5(temp)  # torch.Size([20, 20, 22]) 
        #temp = nn.BatchNorm1d(20)(temp)
        temp = F.relu(temp)  # torch.Size([20, 20,22])
        temp = nn.MaxPool1d(2, stride=2)(temp) # torch.Size([20, 20,11])
        
        out_cnn = self.fc_dropout(temp)
        out = out_cnn.reshape(out_cnn.shape[0],-1)
    
        return out 
    


# In[23]:


"""
卷积核大小6，步数6

与之前的差异主要在Part1，本cnn模型，注重的是分别对左、右、左-右三组的六个变量学习


"""

class BaseCNN3(nn.Module):
    def __init__(self, opts):
        super(BaseCNN3, self).__init__()

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)

        self.channel_num = opts.channel_num
        self.series_size = opts.series_size
        self.feature_dim = opts.feature_dim

        
        self.stride = opts.stride
        self.embed_dropout = opts.embed_dropout
        self.fc_dropout = opts.fc_dropout
        self.len = opts.series_size
        self.out_size = opts.cnn_output_size

        #self.relu = nn.LeakyReLU()
        self.conv = nn.Conv1d(3, 20, (5,6), stride=(1,6)) 
        self.conv2 = nn.Conv1d(20, 20, (5,3), stride=self.stride)
        self.conv3 = nn.Conv1d(20,100, 4, stride=self.stride)
        self.conv4 = nn.Conv1d(100,50,3, stride=self.stride)
        self.conv5 = nn.Conv1d(50, 20, 5,1)       
        self.conv6 = nn.Conv1d(20, 20, 5,1) 
        in_fea = 220
       
        self.fc_dropout = nn.Dropout(self.fc_dropout)

    def forward(self, input1):

        x = Variable(input1)  # torch.Size([20，94,18,3])                 
        x =  x.permute(0,3,1,2)
        
        temp = self.conv(x)   # torch.Size([20，3,94,18]) ==>  torch.Size([20，20，90,3]
        temp = self.conv2(temp)   #torch.Size([20，20，90,3] ==>  torch.Size([20，20，86,1]
        #temp = nn.BatchNorm2d(20)(temp)
        out = F.relu(temp)  
        temp = F.avg_pool2d(out, kernel_size=(2,1))   ## #####  20,20,43,1
       
        temp = self.conv3(temp.squeeze(3))  # 20,50,40
        temp = self.conv4(temp)   # 20,20,38
        #temp = nn.BatchNorm1d(50)(temp)
        temp = F.relu(temp)  # 20,20,38
        temp = nn.MaxPool1d(2, stride=2)(temp) # torch.Size([20,20,19) 
        temp = self.conv5(temp)  # torch.Size([20, 20,15]) 
        temp = F.relu(temp) 
        temp = self.conv6(temp)  #[20, 20,15]
      
        
        out_cnn = self.fc_dropout(temp)
        out = out_cnn.reshape(out_cnn.shape[0],-1)
    
        return out 
    
    


# In[24]:


"""
把延迟嵌入的3维看作3个通道，注重对单一变量内部的信号进行捕获
"""
class BaseCNN4(nn.Module):
    def __init__(self, opts):
        super(BaseCNN4, self).__init__()

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)

        self.channel_num = opts.channel_num
        self.series_size = opts.series_size
        self.feature_dim = opts.feature_dim

        #self.relu = nn.LeakyReLU()
        
        self.stride = opts.stride
        self.embed_dropout = opts.embed_dropout
        self.fc_dropout = opts.fc_dropout
        self.len = opts.series_size
        self.out_size = opts.cnn_output_size

        self.conv = nn.Conv1d(3, 10, (24,1), stride=(2,1)) 
        self.conv2 = nn.Conv1d(10, 50, (3,1), stride=(3,1))
        self.linear = nn.Linear(180, 20) #####8.23
        
       

    def forward(self, input1):

        x = Variable(input1)  # torch.Size([20，94,18,3])                 
        x =  x.permute(0,3,1,2)
        
        temp = self.conv(x)   # torch.Size([20，3,94,18]) ==>  torch.Size([20，10，36,18]
        #temp = nn.BatchNorm2d(10)(temp)
        temp = F.relu(temp)
        temp = F.avg_pool2d(temp, kernel_size=(3,1)) #20,10,12,18
        temp = self.conv2(temp)   #20,50,4,18
        #temp = nn.BatchNorm2d(50)(temp)
        out = F.relu(temp)  
        temp = F.avg_pool2d(out, kernel_size=(4,1))   ## #####  20,50,1,18
         
        temp = temp.permute(0,3,1,2)
        
        temp = F.max_pool2d(temp, kernel_size=(5,1)) #####20,18,50,1 》〉》〉。20，18，10，1
        temp = temp.reshape(temp.shape[0],-1)
        out = self.linear(temp)   ###### 输出20
       
    
        return out 
    


# In[ ]:





# In[25]:



"""
模型设置：单一变量2+stride3版2+原始2+LSTM20-2 bmi

"""

class Multi_Channel_CNN(nn.Module):
    def __init__(self, opts):
        super(Multi_Channel_CNN, self).__init__()

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        self.cnn_num = opts.cnn_num
        final_dim = (self.cnn_num-1) * opts.cnn_output_size +opts.overall_output_size+ opts.lstm_output_size+1
        
        """
        cnn - 原始
        cnn2 - 卷积核3 步长3
        cnn3 - 卷积核6 步长6
        cnn4 - 3 作通道 学习单一变量内部
        
        """
        self.base_lstm = BaseLSTM(opts=opts)
        self.relu = nn.LeakyReLU()
        
        self.base_cnn1 = BaseCNN(opts=opts)
        self.base_cnn2 = BaseCNN2(opts=opts)
        self.base_cnn3 = BaseCNN3(opts=opts)
        self.base_cnn4 = BaseCNN4(opts=opts)
        
        
        self.overall_linear = nn.Linear(20,opts.overall_output_size)
        self.cnn_linear = nn.Linear(220, opts.cnn_output_size)
        self.lstm_linear = nn.Linear(opts.lstm_feature_size, opts.lstm_output_size)

        self.center = CenterLoss(feat_dim=final_dim)
        self.triplet = TripletLoss(margin = opts.triplet_margin)
        
        self.final_liner1 = nn.Linear(final_dim, opts.final_mid_dim)
        self.final_bn = nn.BatchNorm1d(opts.final_mid_dim)
        self.final_liner2 = nn.Linear(opts.final_mid_dim, 1)
        
    def forward(self, input1, input2):

        x = Variable(input1)  # torch.Size([20，94,18,3])                 
        x2 = Variable(input2)
        x_lstm = torch.mean(x, dim=3)
        #x_lstm = torch.norm(x,p=2,dim=3)
        
        x_add = x.permute(2,0,1,3)
        #= x.permute(2,0,1,3)
        index = [0,6,12,1,7,13,2,8,14,3,9,15,4,10,16,5,11,17]
        x_add = x_add[index]
        x_add = x_add.permute(1,2,0,3)
       

        
       ############# MY CNN #########
    
        out = self.base_cnn1(x) #20
        out = self.cnn_linear(out) #####输出2个值
        
        if self.cnn_num > 1:
            out2 = self.base_cnn2(x_add)
            out2 = self.cnn_linear(out2)#####输出2个值
            out = torch.cat([out,out2], dim=1)  
            
        if self.cnn_num >2:
            out3 = self.base_cnn3(x) 
            out3 = self.cnn_linear(out3) 
            out = torch.cat([out,out3], dim=1) 
            
        if self.cnn_num >3:
            out4 = self.base_cnn4(x) 
            out4 = self.overall_linear(out4) 
            out = torch.cat([out,out4], dim=1) 
        

        ##########   lstm  #########

        x_lstm = self.base_lstm(x_lstm) ##输出 lstm/-outputsize
        x_lstm = self.lstm_linear(x_lstm)
        
       
        ###############
        #print(out.shape)
        feature = torch.cat([out,x_lstm,x2], dim=1) 

        #print(feature.shape)
        out = self.final_liner1(feature) 
        out = self.final_bn(out)
        out =  self.relu(out)  # torch.Size([20, 20,22])
        out = self.final_liner2(out) 
        
        return feature,out 
    


# In[26]:



#-----------------triplet_loss 预训练

def model_pre_train_triplet(triplet_margin,epoches,learnRate):
    global batch,opts,val_index,test_index,train_final_index,org_users_feature,org_users_label,org_4_label,org_users_weight
    
    train_num = len(train_final_index)
    
    sample_weight = np.ones(train_num)/train_num   #生成样本初步权重
    sample_weight.reshape(-1,1)
    #print("sample_weight.shape:{}".format(sample_weight.shape))
    
    train_data_batchs = org_users_feature[train_final_index] 
    users_train_label = org_users_label[train_final_index] 
    users_weight = org_users_weight[train_final_index]
    

    #保存后续的基础分类器
    models = []
    base_acc_test = []
    base_AUC_test = []
    for model_id in range(1):

        #编译模型 
        global model,DEBUG
        DEBUG = False
        setOpts(learnRate, opts.cnn_num,triplet_margin=triplet_margin)
        #model = Multi_Channel_CNN(opts=opts)
        
        optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
        
        train_acc = []
        train_loss = []

        epoches = epoches
        batch_size = batch
        
        user_list = [i  for i in range(len(train_data_batchs))]

        BEST_ACCURACY = 0
        BEST_AUC = 0
        BEST_EPOCH = 0

        for epoch in tqdm(range(epoches)):
    #for epoch in range(epoches):
            total_loss = torch.Tensor([0])
            correct_num = 0
            inst_num = 0
            total_step = len(train_data_batchs)
            #print("epoch %d:"%(epoch))
            iteration_num = int(len(train_data_batchs)/batch_size)
            for index in range(iteration_num):
                user_indices = np.random.choice(user_list, batch_size,sample_weight.tolist()).tolist()
                #print("user_indices:{}\n".format(user_indices))
                users_feature_batch = train_data_batchs[user_indices]
                users_label_batch = users_train_label[user_indices]
                users_weight_batch = users_weight[user_indices]
                sample_weight_batch = sample_weight[user_indices]
            
                model.train()
                optimizer.zero_grad()

                inst_num += 1
                features = Variable(torch.FloatTensor(users_feature_batch))
                weights_feature = Variable(torch.FloatTensor(users_weight_batch))
                label = Variable(torch.FloatTensor(users_label_batch))
  
                if opts.use_cuda:
                    features = features.cuda()
                    label = label.cuda()
                
                feature,pred = model(features,weights_feature)           
                loss2 = torch.nn.BCEWithLogitsLoss()(pred, label.reshape(-1,1))
                loss1 = model.triplet(feature,label)
                #loss1 = model.triplet(feature,label).long()
                
                loss = loss1 + loss2
                
                loss1.backward(retain_graph=True) 
                loss2.backward()  
                
                optimizer.step()
                #loss = loss1

                loss = loss.cpu()
                total_loss += loss.data

            train_loss.append(total_loss.numpy())
        pict_loss(epoches,train_loss)
    


# In[27]:



#-----------------打印验证集合上的效果
def model_pre_train_center(cnn_num,mid_dim,epoches,learnRate):
    global  batch,val_index,test_index,train_final_index,org_users_feature,org_users_label,org_4_label,org_users_weight
    
    train_num = len(train_final_index)
    
    sample_weight = np.ones(train_num)/train_num   #生成样本初步权重
    sample_weight.reshape(-1,1)
    #print("sample_weight.shape:{}".format(sample_weight.shape))
    
    train_data_batchs = org_users_feature[train_final_index] 
    users_train_label = org_users_label[train_final_index] 
    users_weight = org_users_weight[train_final_index]

    for model_id in range(1):

        #编译模型 
        global model,DEBUG
        DEBUG = False
        setOpts(learnRate, cnn_num,final_mid_dim=mid_dim)
        model = Multi_Channel_CNN(opts=opts)
        
        optim_center = optim.SGD(model.center.parameters(), lr=0.2, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
        
        train_acc = []
        train_loss = []

        epoches = epoches
        batch_size = batch
        
        user_list = [i  for i in range(len(train_data_batchs))]

        BEST_ACCURACY = 0
        BEST_AUC = 0
        BEST_EPOCH = 0

        for epoch in tqdm(range(epoches)):
    #for epoch in range(epoches):
            total_loss = torch.Tensor([0])
            correct_num = 0
            inst_num = 0
            total_step = len(train_data_batchs)
            #print("epoch %d:"%(epoch))
            iteration_num = int(len(train_data_batchs)/batch_size)
            for index in range(iteration_num):
                user_indices = np.random.choice(user_list, batch_size,sample_weight.tolist()).tolist()
                #print("user_indices:{}\n".format(user_indices))
                users_feature_batch = train_data_batchs[user_indices]
                users_label_batch = users_train_label[user_indices]
                users_weight_batch = users_weight[user_indices]
                sample_weight_batch = sample_weight[user_indices]
            
                model.train()
                optimizer.zero_grad()

                inst_num += 1
                features = Variable(torch.FloatTensor(users_feature_batch))
                weights_feature = Variable(torch.FloatTensor(users_weight_batch))
                label = Variable(torch.FloatTensor(users_label_batch)).squeeze(1)
  
                if opts.use_cuda:
                    features = features.cuda()
                    label = label.cuda()
                
                feature,pred = model(features,weights_feature)           
                loss2 = torch.nn.BCEWithLogitsLoss()(pred, label.reshape(-1,1))
                loss1 = model.center(feature.float(),label.float()) 
                
                loss = loss1 + loss2
                
                loss1.backward(retain_graph=True)
                optim_center.step()
                
                loss2.backward()            
                optimizer.step()

                loss = loss.cpu()
                total_loss += loss.data

            train_loss.append(total_loss.numpy())

        pict_loss(epoches,train_loss)
    


# In[28]:


######----------正式训练------(Adam) 设置batch
def model_train(error_min,model_num, epoches,learnRate,threshold):
    global fail_lists,cor_lists,batch,p_prototype,val_index,test_index,train_final_index,org_users_feature,org_users_label,org_4_label,org_users_weight
    
    train_num = len(train_final_index)
    
    sample_weight = np.ones(train_num)/train_num   #生成样本初步权重
    sample_weight.reshape(-1,1)
    #print("sample_weight.shape:{}".format(sample_weight.shape))
    
    train_data_batchs = org_users_feature[train_final_index] 
    users_train_label = org_users_label[train_final_index] 
    users_weight = org_users_weight[train_final_index]
    
   # print("users_train_label.shape:{}".format(users_train_label.shape))
    # users_4_label = org_4_label[train_final_index]

    #生成基分类器 系数数组
    my_alpha = [] 
    
    #保存后续的基础分类器
    models = []
    base_acc_test = []
    base_AUC_test = []
    for model_id in range(20):
        
        if len(my_alpha)== model_num:
            break
            
        sum_sample_weight = sum(sample_weight)
        sample_weight = sample_weight / sum_sample_weight
        #print("sample to list:{}".format(sample_weight.tolist()))
        
        print("训练集大小：{}".format(len(train_data_batchs)))
        shuffle = True

        #编译模型 
        global model
        #setOpts(learnRate)
        #model = Multi_Channel_CNN(opts=opts)

        #optim_center = optim.SGD(model.center.parameters(), lr=0.2, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
        #optimizer = optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9)
        train_acc = []
        train_loss = []

        epoches = epoches
        batch_size = batch
        
        user_list = [i  for i in range(len(train_data_batchs))]

        BEST_ACCURACY = 0
        BEST_AUC = 0
        BEST_EPOCH = 0
        global DEBUG 
        DEBUG = False

        for epoch in tqdm(range(epoches)):
    #for epoch in range(epoches):
            total_loss = torch.Tensor([0])
            correct_num = 0
            inst_num = 0
            total_step = len(train_data_batchs)
            #print("epoch %d:"%(epoch))
            iteration_num = int(len(train_data_batchs)/batch_size)
            for index in range(iteration_num):
                user_indices = np.random.choice(user_list, batch_size,sample_weight.tolist()).tolist()
                #print("user_indices:{}\n".format(user_indices))
                users_feature_batch = train_data_batchs[user_indices]
                users_label_batch = users_train_label[user_indices]
                users_weight_batch = users_weight[user_indices]
                sample_weight_batch = sample_weight[user_indices]
            
                model.train()
                optimizer.zero_grad()

                inst_num += 1
                features = Variable(torch.FloatTensor(users_feature_batch))
                weights_feature = Variable(torch.FloatTensor(users_weight_batch))
                label = Variable(torch.FloatTensor(users_label_batch))
                coefficient = torch.Tensor(sample_weight_batch.reshape(-1,1))
                #print("coefficient.shape:{}".format(coefficient.shape))
  
                if opts.use_cuda:
                    features = features.cuda()
                    label = label.cuda()
            
                feature,pred = model(features,weights_feature)           
                loss = torch.nn.BCEWithLogitsLoss()(pred, label.reshape(-1,1))
                #loss2 = torch.nn.CrossEntropyLoss()(pred, label.long())
                #loss1 = model.center(feature,label.squeeze()) 
                #loss3 = model.triplet(feature,label)
                #loss = loss1 + loss2
                
                #loss1.backward(retain_graph=True)
                loss.backward()
                #loss3.backward(retain_graph=True)
                
                #optim_center.step()
                optimizer.step()

                loss = loss.cpu()
                total_loss += loss.data

            train_loss.append(total_loss.numpy())
        
            accracy,precision,recall,F1_score,EER,AUC,check= evaluation(threshold)
        
            train_acc.append(accracy)
            model_path = "models/bmi/final_"+str(model_id)+".pth"
            
            if epoch > 15:
                if AUC > BEST_AUC:
                    BEST_AUC = AUC
                    BEST_ACCURACY = accracy
                    BEST_EPOCH = epoch
                    torch.save(obj=model.state_dict(), f="models/bmi/final_"+str(model_id)+".pth")
                    
                elif AUC == BEST_AUC:
                    if accracy >= BEST_ACCURACY:
                        BEST_ACCURACY = accracy
                        BEST_EPOCH = epoch
                        torch.save(obj=model.state_dict(), f="models/bmi/final_"+str(model_id)+".pth")
                    
        #pict_loss(epoches,train_loss)
        #检查训练集效果
        error,pred_check = evaluate_train_ada(threshold,train_final_index,model_path)
        
        if error > error_min and len(my_alpha) >= 2:
            break
        elif error > error_min and len(my_alpha) < 2 :
            continue
        elif error ==0:
            error = error+0.00001
            
        if BEST_ACCURACY < 1-error_min:
            continue 
            
        #添加基础分类器系数
        alpha = 0.5*np.log((1 - error)/error)
        my_alpha.append(alpha)
        
        #更新sample_weight
        change = -2*(np.array(pred_check).reshape(-1, 1) - 0.5)
        #print("change.shape:{}".format(change.shape))
        change_a = change * alpha
        #print("change_s.shape:{}".format(change_a.shape))
        sample_weight = (sample_weight.reshape(-1,1) * np.exp(change_a))
        #print("更新sample_weight.shape:{}".format(sample_weight.shape))
        
        print("-------------")
        print("第{}个基分类器：alpha={}  path: {}".format(len(my_alpha), alpha,model_path))
        models.append(model_path)
        
        #在测试集上跑一下
        pred_model_list = test_base_model(threshold,model_path)
        accracy,precision,recall,F1_score ,test_EER,test_AUC,check= test(threshold,pred_model_list)
        base_acc_test.append(accracy)
        base_AUC_test.append(test_AUC)
        print("train-acc: {}   ".format(1-error))
        print("val-acc: {}   val-AUC:{} ".format(BEST_ACCURACY,BEST_AUC))
        print("test-acc: {}   test-AUC:{} ".format(accracy,test_AUC))
        print("-------------")
    final_predict = weighted_vote(test_index, models, my_alpha, threshold)

    accracy,precision,recall,F1_score ,test_EER,test_AUC,check= test(threshold,final_predict)
    check = np.array(check)
    #print("check:{}".format(check))
    fail_index = np.where(check==0)#######取错的索引
    cor_index = np.where(check==1)
    #print("fail_index:{}".format(fail_index[0]))
    
    fail_list = np.array(test_index)[fail_index[0]]
    cor_list = np.array(test_index)[cor_index[0]]
    fail_lists.append(fail_list)
    cor_lists.append(cor_list)
    
    ada_path = 'adaboost_models/adaboost_{}.pkl'.format(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    print("adaboost_path: {}".format(ada_path))
    pickle.dump(models,open(ada_path,'wb+'))

    
    return base_acc_test,base_AUC_test,accracy,precision,recall,F1_score,test_EER,test_AUC,fail_list
    


# In[ ]:


#模型设置：单一变量2+stride3版2+原始2+LSTM2  
#CNN数量：3 分类


#调节模型参数
class Opts:
    pass
def setOpts(learnRate,cnn_num,triplet_margin=0.3,final_mid_dim=5):
    global opts, org_users_feature 
    opts = Opts()
    opts.seed = 20
    opts.stride = 1
    opts.channel_num = opts.embed_size = 100         #原本参数为100，但由于训练数据有限，为避免过拟合/欠拟合，将该参数降低
                                                    ##每条用户数据映射出的向量维度opts.embed_size， 同理kernel_num
    opts.series_size = org_users_feature.shape[1]  ##序列长度
    opts.embed_dropout = 0.3
    opts.fc_dropout = 0.3
    opts.triplet_margin = 0.3
    opts.final_mid_dim = final_mid_dim
    
    opts.overall_output_size =2
    
    opts.feature_dim = org_users_feature.shape[2]         #每条用户数据原始的向量维度， 自动识别维度
    opts.lr = learnRate
    opts.lr_decay_rate = 0.5
    opts.weight_decay = 1e-8
    opts.use_cuda = False
    
    opts.cnn_output_size = 2
    opts.cnn_feature_size = 220
    opts.cnn_num  = cnn_num
    
    opts.lstm_pool_size = 5  ###lstm 池化大小
    opts.lstm_output_size = 2
    opts.lstm_feature_size = 10
    opts.lstm_hidden_size = 20
    opts.lstm_num_layer = 2

batch = 20
loadSavedData()

fail_lists = []
cor_lists = []

all_acc_list = []
all_precision_list = []
all_recall_list = []
all_F1_list = []
all_EER_list = []
all_AUC_list = []

for i in range(10):
    k=1
    
    test_acc_list = []
    test_precision_list = []
    test_recall_list = []
    test_F1_list = []
    test_EER_list = []
    test_AUC_list = []
    for train_index, test_index in get_k_fold(5,org_users_feature,seed=None):
        print("-----第{}折----".format(k))

        k = k+1
        makeDataEqual(train_index,test_index)
        changeRatio(1,1)
        randomTrainData()
        #batch =len(train_final_index)
       # batch=20
        model_pre_train_center(3,10,20,0.01)  #cnn数量、特征嵌入的隐藏层维度，
       # pict_pca()
        model_pre_train_triplet(3,25,0.001)
        #pict_pca()
        base_acc_test,base_AUC_test,accracy,precision,recall,F1_score,test_EER,test_AUC,fail_list = model_train(0.5,1,30,0.01,0.5)


        test_acc_list.append(accracy)

        test_precision_list.append(precision)
        test_recall_list.append(recall)
        test_F1_list.append(F1_score)
        test_EER_list.append(test_EER)
        test_AUC_list.append(test_AUC)

        print("------{}折结果-------".format(k-1))
        print("TEST_ACC：{:.3f}".format(accracy))
        print("TEST_PRECISION：{:.3f}".format(precision))
        print("TEST_RECALL：{:.3f}".format(recall))
        print("TEST_F1：{:.3f}".format(F1_score))
        print("TEST_EER：{:.3f}".format(test_EER))
        print("TEST_AUC：{:.3f}".format(-test_AUC))
        print("----------------------")

    test_acc = np.mean(test_acc_list)
    precision = np.mean(test_precision_list)
    recall = np.mean(test_recall_list)
    F1_score = np.mean(test_F1_list)
    test_EER = np.mean(test_EER_list)
    test_AUC = np.mean(test_AUC_list)

    print("------第{}次平均结果-------".format(i+1))
    print("TEST_ACC：{:.3f}".format(test_acc))
    print("TEST_PRECISION：{:.3f}".format(precision))
    print("TEST_RECALL：{:.3f}".format(recall))
    print("TEST_F1：{:.3f}".format(F1_score))
    print("TEST_EER：{:.3f}".format(test_EER))
    print("TEST_AUC：{:.3f}".format(-test_AUC))
    print("----------------------")
    all_acc_list.append(test_acc)
    all_precision_list.append(precision)
    all_recall_list.append(recall)
    all_F1_list.append(F1_score)
    all_EER_list.append(test_EER)
    all_AUC_list.append(-test_AUC)

  
    
print("------十次平均结果-------")
print("TEST_ACC：{:.3f}".format(np.mean(all_acc_list)))
print("TEST_PRECISION：{:.3f}".format(np.mean(all_precision_list)))
print("TEST_RECALL：{:.3f}".format(np.mean(all_recall_list)))
print("TEST_F1：{:.3f}".format(np.mean(all_F1_list)))
print("TEST_EER：{:.3f}".format(np.mean(all_EER_list)))
print("TEST_AUC：{:.3f}".format(np.mean(all_AUC_list)))
print("----------------------")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




