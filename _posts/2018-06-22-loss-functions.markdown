---    
layout: post
title:  "Loss Functions"
date:   2018-06-22 10:00:00
categories: MachineLearning
tags: MachineLearning
excerpt: 
---

通常机器学习的目标函数是损失函数 + 正则项的形式

例如如下是一种均方误差 (MSE) + 正则项的形式，L是损失函数，$$ \Phi $$是正则项：

$$ \theta^* = \arg \min_\theta \frac{1}{N}{}\sum_{i=1}^{N} L(y_i, f(x_i; \theta)) + \lambda\  \Phi(\theta) $$

下面列举几种常见的损失函数

### Zero-one Loss

Zero-one Loss即0-1损失，它是一种较为简单的损失函数，如果预测值与目标值不相等，那么为1，否则为0。

### 平方损失函数（square loss）

平方损失形式如下，常用在回归问题中，也可以用作二分类：

$$ l(y,f(x))=(y - f(x))^2 $$

### Hinge Loss

Hinge损失可以用来解决间隔最大化问题，如SVM。

$$ l(y, f(x)) = max(0, 1 - y * f(x)) $$，其中$$ y \in \{-1, 1\} $$

其中f(x)是原始输出而非label。

例如$$ y=1 $$时，下图中，$$ f(x) >= 1 $$时，$$ l=0 $$, 否则$$ l $$线性增大。

<img src="/images/loss_func/1.png" width="40%" height="40%">


### Logistic loss

逻辑回归中常见的损失函数：

$$ l = log(1+e^{-y w \cdot x}) $$，其中$$ y \in \{-1, 1\} $$

### Cross entropy loss (Log Loss)

也是逻辑回归中常见的损失函数，与上面的区别是$$ y \in \{0, 1\} $$


\begin{align}   
H(p,q) &=-\sum\limits_i p_i log{q_i} \\\
&= -ylog{\hat y} - (1-y)log{(1-\hat y)}
\end{align}

其中p是交叉熵中的真实label分布，q是当前模型的预测值的分布。

### 正则项

再简单说一下正则项：

常见的正则项有L1和L2，是为了避免过拟合。

过拟合的时候，拟合函数的系数往往非常大，过拟合，就是拟合函数需要顾及每一个点，
最终形成的拟合函数波动很大。在某些很小的区间里，函数值的变化很剧烈。
这就意味着函数在某些小区间里的导数值（绝对值）非常大，由于自变量值可大可小，所以只有系数足够大，才能保证导数值很大。

L1是所有权重w的绝对值的和，乘以$$ \frac \lambda m $$。
L2是所有权重w的绝对值的平方和，乘以$$ \frac \lambda m $$。

L1优点是能获取稀疏的模型，缺点是加入L1后目标函数在原点不可导。

对于L2，它的效果是减小w，这也就是权重衰减（weight decay）的由来，
而对于L1，它的效果就是让w往0靠，使网络中的权重尽可能为0，也就相当于减小了网络复杂度，防止过拟合。

下面说一下为什么L1可以实现稀疏性；L2可以选更多的参数，并且在0附近。

<img src="/images/loss_func/2.png" width="40%" height="40%">

对于L1-正则项来说，因为L1-正则项的等值线是一组菱形，
这些交点容易落在坐标轴上。因此，另一个参数的值在这个交点上就是零，从而实现了稀疏化。

而L2-正则项的等值线是一组圆形，交点落在坐标轴上的概率则不大，但是集中在0附近。

也可以看出L1在特征选择时很有用，而L2则仅是正则化的作用。

### 参考资料

[https://en.wikipedia.org/wiki/Loss_functions_for_classification](https://en.wikipedia.org/wiki/Loss_functions_for_classification)

[https://cloud.tencent.com/developer/article/1047645](https://cloud.tencent.com/developer/article/1047645)

[https://blog.csdn.net/zouxy09/article/details/24971995](https://blog.csdn.net/zouxy09/article/details/24971995)

[https://blog.csdn.net/u012162613/article/details/44261657](https://blog.csdn.net/u012162613/article/details/4426165)

[https://liam0205.me/2017/03/30/L1-and-L2-regularizer/](https://liam0205.me/2017/03/30/L1-and-L2-regularizer/)
