#!/usr/bin/env python
# coding: utf-8

# In[43]:


#数据预处理
import numpy as np
import DaPy as dp
from random import randint, random


# In[45]:


data = dp.read('Updates_NC.csv').data

def decode(val, code='utf-8'): 
    if isinstance(val,bytes):
        return val.decode(code) 
    return val 
data = data.map(decode, cols=['报道时间', '省份', '城市', '消息来源'], inplace=True)


# In[ ]:


#用23日之前的到10日左右能找到的只有6天数据，尝试过但效果不佳，所以决定加入封城之后的数据；这里假设以16号作为初始；
#2月12日诊断方式发生了改变，数目激增，所以决定模型的训练数据从1月17日到2月11日，当然这段时间之内存在不同强度的措施，这里只是假设一样。
wuhan = data.query(' 城市 == "武汉市" and 报道时间 >"1月16日"').reverse()
wuhan = wuhan.groupby('报道时间', max, apply_col=['新增确诊', '新增出院', '新增死亡'])
#wuhan.insert_row(3, ['1月14日', 0, 0, 0]) # 补充数据,找不到
#wuhan.insert_row(4, ['1月15日', 0, 0, 0]) # 补充数据，找不到
#wuhan.insert_row(0, ['1月16日', 45, 2, 15])#截止16日累计数据
wuhan.insert_row(0, ['1月17日累计', 62, 6, 15]) # 累计的补充国家卫检委数据
wuhan.insert_row(6, ['1月23日', 62, 0, 17]) # 补充国家卫检委数据 
wuhan.show()


# In[47]:


from scipy.optimize import dual_annealing, minimize
from sklearn.metrics import r2_score
from collections import namedtuple 
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']


# In[48]:


SEIDC_PARAM = namedtuple('SEIDCparm', ['alpha1', 'alpha2', 'beta', 'sigma', 'gamma'])

class SEIDC(object):
    def __init__(self, P=None): 
        self.P = P
     
    #估计值定义，上述公式，返回S,E,I,D,C
    def _forward(self, S, E, I, D, C, param, max_iter):
        a1, a2, b, s, g=param
        est = dp.Table(columns=['S','E','I','D','C'])
        for t in range(max_iter):
            S_ = S - a1 * E-a2 * I+s * I
            E_ = E  + a1 * E+ a2 * I-b *E
            I_ = I + b * E - s * I - g * I
            D_ = D+ g * I
            C_ = C +s * I
            S, E, I, D, C = S_, E_, I_, D_, C_
            est.append_row([S, E, I, D, C])
        return est 
   
   #定义损失函数loss
    def _loss(self, obs, est):
        assert len(obs) == len(est)
        loss = ((np.log2(obs + 1) - np.log2(est + 1)) ** 2).sum()
        self.lossing.append(loss)
        return loss
    
    def _optimize(self, param, s, e, i, d, c, obs):
        est = self._forward(s, e, i, d, c, param, len(obs))
        return self._loss(obs, est['I', 'D', 'C'].toarray())
     
        #dual_annealing退火优化方法参数说明，使得self._optimize最小，param参数的取值范围，args对应目标函数的可变参数
    def fit(self, initS, initE, initI, initD, initC, Y):
        self.lossing = []
        args = (initS, initE, initI, initD, initC, Y['确诊', '死亡', '治愈'].toarray())
        param = [(0, 1),] * 5
        result = dual_annealing(self._optimize, param, args=args, seed=30, maxiter=10) ['x']
        self.P = SEIDC_PARAM(*result)
     
    #score函数  ，R2是拟合优度，越接近于1越号，用的是r2_score函数 
    def score(self, initS, initE, initI, initD, initC, Y, plot=False):
        est = self.predict(initS, initE, initI, initD, initC, len(Y))['I', 'D', 'C']
        loss = self._loss(Y['确诊', '死亡', '治愈'].toarray(), est.toarray())
        est.columns = ['确诊', '死亡', '治愈']
        r1 = r2_score(Y['治愈'], est['治愈'])
        r2 = r2_score(Y['死亡'], est['死亡'])
        r3 = r2_score(Y['确诊'], est['确诊'])
        if plot:
            self.plot_predict(Y, est)
            print(' - 平均潜伏期为：%.2f天' % (1.0 / self.P.beta))
            print(' - 病毒再生基数：%.2f' % (self.P.alpha1 / self.P.beta + (self.P.alpha2 / self.P.sigma + self.P.alpha2 / self.P.gamma)/ 2))
            print(' - 确诊R2：%.4f' % r3)
            print(' - 死亡R2：%.4f' % r2)
            print(' - 治愈R2：%.4f' % r1)
            print(' - 模型R2：%.4f' % ((r1 + r2 + r3) / 3))
            print(' - 模型总误差：%.4f' % loss)
        return loss, (r1 + r2 + r3) / 3
        
    def plot_error(self):
        plt.plot(self.lossing, label=u'正确率')
        plt.legend()
        plt.show()
        
    def plot_predict(self, obs, est):
        for label, color in zip(obs.keys(), ['red', 'black', 'green']):
            plt.plot(obs[label], color=color,alpha=0.3,label=label)
            plt.plot(est[label], color=color, alpha=0.6, label=label)
            plt.legend()
            plt.show()
            
    def predict(self, initS, initE, initI, initD,initC, T):
        return self._forward(initS, initE, initI, initD, initC, self.P, T)
 


# In[ ]:


train = wuhan['新增确诊', '新增死亡', '新增出院']
train.columns = ['确诊', '死亡', '治愈']
train.accumulate(inplace=True)


# In[ ]:


# 截止到1月16日的累计确诊45、治愈2、死亡人数15，0到1100是参考的，可以换成更大的试试

def searchBestParam(seir):
    min_loss, max_r2, best_param, likeli_potential = float('inf'),0.0,None, 0
    for potential in range(0, 1100, 25):
        seir.fit(900000, potential, 45,15,2,train)
        loss, r2 = seir.score(9000000, potential,45,15,2, Y=train, plot=False)
        if loss < min_loss and r2 > max_r2:
            print('潜在患者：%.4f | R2：%.4f | 误差： %.6f' % (potential, r2, loss))
            min_loss, max_r2, best_param, likeli_potential = loss, r2, seir.P, potential
    seir.P = best_param
    seir.score(9000000, likeli_potential, 45,15,2, Y=train, plot=True)
    return seir, likeli_potential

seir, potentials = searchBestParam(SEIDC())

