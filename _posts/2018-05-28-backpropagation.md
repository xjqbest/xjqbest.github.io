---
layout: post
title:  "反向传播"
date:   2018-05-28 11:00:01
categories: MachineLearning
tags: MachineLearning
excerpt: 
---

## Backpropagation

反向传播（Backpropagation，简称BP）,是一种用来训练神经网络的常见方法。该方法对网络中所有权重计算损失函数的梯度，进而更新权重以最小化损失函数。

假设有样本集$$ {(x^{(1)},y^{(1)},...,(x^{(m)},y^{(m)})} $$，包含m个样本，可以使用批量批度下降的方法求解神经网络。设损失函数为：
\begin{align}
J(W,b;x,y) = \frac 12 (h_{W,b}(x)-y)^2
\end{align}

对于m个样本，可以有如下定义,其中第一项是平方误差，第二项是正则项（也叫权重衰减项）：
\begin{align}
J(W,b) = [\frac 1m \sum_{i=1}^{m}{J(W,b;x^{(i)},y^{(i)}}] + \frac \lambda 2 \sum_{l=1}^{n_l-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} {W_{ij}^{(l)}}^2
\end{align}

我们的目标是针对参数$$ W $$和$$ b $$，来求解函数$$ J(W,b) $$的最小值。初始化时需要将这两个参数初始化为很小的、接近0的值。如果所有参数都用相同的值作为初始值，那么所有隐藏层单元最终会得到与输入值有关的、相同的函数。

梯度下降法中每一次迭代都按照如下公式对参数$$ W $$和$$ b $$更新。其中 $$ \alpha $$是学习率：
\begin{align}
W_{ij}^{(l)} &= W_{ij}^{(l)} - \alpha \frac {\partial}{\partial W_{ij}^{(l)}}J(W,b)  \\\
b_i^l &= b_i^l - \alpha \frac {\partial}{\partial b_i^l}J(W,b)
\end{align}


算法步骤：  
1. 进行前向传播计算，得到$$ l_2 , l_3 ,..., l_{n_l} $$的输出值。  
2. 对于第$$ n_l $$层的每个输出单元i，根据如下公式计算残差：  
\begin{align}
\delta_i^{n_l} &= \frac {\partial}{\partial z_i^{n_l}} \frac 12 (y - h_{W,b}(x))^2 \\\
&=-(y_i - a_i^{(n_l)}) * f'(z_i^{(n_l)})
\end{align}
其中$$ a_i^{(n_l)} = f'(z_i^{(n_l)}) $$，即f是激活函数。
3. 对于$$ l=n_l-1,n_l-2,...,2 $$的各个层，第l层的第i个节点的残差计算方法如下：  
\begin{align}
\delta_i^{(l)} = (\sum_{j=1}^{ s_{l+1}}{W_{ij}^{(l)} \delta_j^{(l+1)} }) * f'(z_i^{(l)})
\end{align}
4. 求出偏导数
\begin{align}
\frac {\partial}{\partial W_{ij}^{(l)}}J(W,b;x,y) &= a_j^{(l)}\delta_i^{(l+1)}  \\\
\frac {\partial}{\partial b_i^l}J(W,b;xmy) &= \delta_i^{(l+1)}
\end{align}

## 参考资料

[反向传导算法](http://ufldl.stanford.edu/wiki/index.php/%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E7%AE%97%E6%B3%95)  
[深度神经网络（DNN）模型与前向传播算法](https://www.cnblogs.com/pinard/p/6418668.html)  
[深度神经网络（DNN）反向传播算法(BP)](https://www.cnblogs.com/pinard/p/6422831.html)