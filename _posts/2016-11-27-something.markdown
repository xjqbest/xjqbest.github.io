---
layout: post
title:  "机器学习一些概念"
date:   2016-11-27 15:34:01
categories: MachineLearning
tags: MachineLearning
excerpt: 记录自己学习中遇到的机器学习的概念，不定期更新，随时记录
---

# 一些概念

## （1）最大后验估计和最大似然估计

最大后验估计（maximum a posteriori probability estimate, 简称MAP）  
最大似然估计（maximum-likelihood estimation, 简称MLE）

假设得到某些观察数据 x，然后需要根据x ，估计未知的总体参数$$ \theta $$  
1. 对于MAP来说，$$ \theta $$是已知先验概率密度函数$$ P(\theta) $$的随机变量  
\begin{align} 
\hat \theta = \mathop{argmax}\limits_{\theta} P(x|\theta)P(\theta)
\end{align}
2. 对于MLE来说，$$ \theta $$是非随机变量或者分布未知的随机变量，这两种情况都可以认为$$ P(\theta) $$是均匀分布的，即P(θ)=C  
\begin{align}
\hat \theta &= \mathop{argmax}\limits_{\theta} P(x|\theta)C \\\
&= \mathop{argmax}\limits_{\theta} P(x|\theta)
\end{align}

## （2）cross-entropy error function
交叉熵(cross entropy)可以用来定义机器学习中的损失函数，$$ p_i $$是真实label的概率，$$ q_i $$是当前模型的预测值的分布。  
对于二分类逻辑回归，给定输入vector$$ \ x $$，输出$$ y \in \{0,1\} $$，则有  
\begin{align}
q_{y=1} &= \hat y = g(w\cdot x)，(其中g(z)=1/(1+e^{-z})) \\\
q_{y=0} &= 1 - \hat y  \\\
\end{align}
真实的概率可以表示为：
\begin{align}
p_{y=1} &=y \\\
p_{y=0} &=1-y
\end{align}
也即：$$ p \in \{y,1-y\}，q \in \{\hat y, 1- \hat y\}$$  
所以p,q之间的交叉熵为  
\begin{align}
H(p,q) &=-\sum\limits_i p_i log{q_i} \\\
&= -ylog{\hat y} - (1-y)log{(1-\hat y)}
\end{align}

另外，逻辑回归的极大似然等价于最小化交叉熵.

## 最小风险贝叶斯决策(bayes conditional risk)
在决策中，除了关心决策的正确与否，有时我们更关心错误的决策将带来的损失。  
考虑各种错误造成损失不同时的一种最优决策，就是最小风险贝叶斯决策。  
设对于实际状态为wj的向量x采取决策αi所带来的损失为
\begin{align}
\lambda (\alpha_i,w_j),i=1,...,k,   j=1,...,c
\end{align}
该函数称为损失函数，通常它可以用表格的形式给出，叫做决策表。
计算步骤如下：  
1. 利用贝叶斯公式计算后验概率：
\begin{align}
P(w_j|x)=\frac{P(x|w_j)P(w_j)}{\sum_{i=1}^{c}p(x|w_i)P(w_i)}，j=1,...,c
\end{align}
2. 利用决策表计算条件风险：
\begin{align}
R(\alpha_i|x)=\sum_{j=1}^{c}\lambda(\alpha_i|w_j)P(w_j|x)，i=1,... k
\end{align}
3. 选择风险最小的决策：
\begin{align}
\alpha=\mathop{argmin}\limits_{\alpha}R(\alpha_i|x)，i=1,...,k
\end{align}

对不同类判决的错误风险一致时，最小风险贝叶斯决策就转化成最小错误率贝叶斯决策。
最小错误贝叶斯决策可以看成是最小风险贝叶斯决策的一个特例。

## 参考资料
[最大似然估计（MLE）和最大后验概率（MAP）](http://blog.csdn.net/upon_the_yun/article/details/8915283)  
[最大后验估计和最大似然估计](http://www.cnblogs.com/emituofo/archive/2011/12/02/2271410.html)  
[Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)