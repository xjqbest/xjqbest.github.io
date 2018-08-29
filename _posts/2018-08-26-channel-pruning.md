---
layout: post
title:  "模型压缩之Channel Pruning"
date:   2018-08-26 14:00:00
categories: MachineLearning
tags: MachineLearning
excerpt: Channel Pruning
---

# 一、简介

[论文地址](/docs/channel_pruning/channel_pruning_paper.pdf)

channel pruning是指给定一个CNN模型，去掉卷积层的某几个输入channel以及相应的卷积核，
并最小化裁剪channel后与原始输出的误差。

可以分两步来解决：
1. channel selection  
利用LASSO回归裁剪掉多余的channel，求出每个channel的权重，如果为0即是被裁减。
2. feature map reconstruction  
利用剩下的channel重建输出，直接使用最小平方误差来拟合原始卷积层的输出，求出新的卷积核W。

# 二、优化目标

## 2.1 定义优化目标

输入c个channel，输出n个channel，卷积核W的大小是$$ n \times c \times k_h \times k_w $$

我们对输入做了采样，假设对每个输入，对channel采样出来N个大小为$$ c \times k_h \times k_w $$块

为了把输入channel从c个剪裁到c'个，我们定义最优化的目标为

<img src="/images/channel_pruning/1.png" width="35%" height="35%">

其中$$ \beta $$是每个channel的权重向量，如果$$ \beta_i $$是0，意味着裁剪当前channel，相应的$$ W_i $$也被裁减。


## 2.2 求最优目标

为了最优化目标，分为如下两步

### 2.2.1 固定W，求$$ \beta $$  

<img src="/images/channel_pruning/2.png" width="45%" height="45%">

其中$$ Z_i = X_i W_i^\mathrm T $$，大小是$$ N \times n$$，

这里之所以加上关于$$ \beta $$的L1正则项，是为了避免所有的$$ \beta_i $$都为1，而是让它们趋于0。

### 2.2.2 固定$$ \beta $$，求W

利用剩下的channel重建输出，直接求最小平方误差

<img src="/images/channel_pruning/3.png" width="27%" height="27%">

其中$$ X' = [\beta_1X_1, \beta_2X_2, \dots \beta_cX_c] $$，大小为$$ N \times ck_hk_w $$，
W'也被reshape为$$ n \times ck_hk_w $$。

### 2.2.3 多分支的情况

论文只考虑了常见的残差网络，设residual分支的输出为$$ Y_2 $$，shortcut 分支的输出为$$ Y_1 $$。

这里首先在residual分支的第一层前做了channel采样，从而减少计算量（训练过程中做的）。

设$$ Y_1' $$为原始的上一层的输出，
那么channel pruning中，residual分支的输出拟合$$ Y_1 + Y_2 - Y_1' $$，其中$$ Y_1' $$是裁减后的shortcut。

<img src="/images/channel_pruning/4.png" width="43%" height="43%">


# 三、实现

实现的时候，不是按照不断迭代第一步和第二步，因为比较耗时。
而是先不断的迭代第一步，直到裁剪剩下的channel个数为c'，然后执行第二步求出最终的W。

## 3.1 第一步Channel Selection

如何得到LASSO回归的输入：

（1）首先把输入做转置  
```python
# (N, c, hw) --> (c, N, hw)
inputs = np.transpose(inputs, [1, 0, 2])
```
（2）把weigh做转置  
```python
# (n, c, hw) --> (c, hw, n)
weights = np.transpose(weights, [1, 2, 0]))
```
（3）最后两维做矩阵乘法  
```python
# (c, N, n), matmul apply dot on the last two dim
outputs = np.matmul(inputs, weights)
```
（4）把输出做reshape和转置
```python
# (Nn, c)
outputs = np.transpose(outputs.reshape(outputs.shape[0], -1))
```

LASSO回归的目标值即是对应的Y，大小为$$ N \times n $$


$$ \lambda $$的大小影响了最终$$ \beta $$为0的个数，为了找出合适的$$ \lambda $$，需要尝试不同的值，直到裁剪剩下的channel个数为$$ c' $$为止。

为了找到合适的$$ \lambda $$可以使用二分查找，
或者不断增大$$ \lambda $$直到裁剪剩下的channel个数$$ \ge c' $$，然后降序排序取前$$ c' $$个$$ \beta_i $$，剩下的$$ 
\beta $$为0。

```python
while True:
    coef = solve(alpha)
    if sum(coef != 0) < rank:
        break
    last_alpha = alpha
    last_coef = coef
    alpha = 4 * alpha + math.log(coef.shape[0])
if not fast:
    # binary search until compression ratio is satisfied
    left = last_alpha
    right = alpha
    while True:
        alpha = (left + right) / 2
        coef = solve(alpha)
        if sum(coef != 0) < rank:
            right = alpha
        elif sum(coef != 0) > rank:
            left = alpha
        else:
            break
else:
    last_coef = np.abs(last_coef)
    sorted_coef = sorted(last_coef, reverse=True)
    rank_max = sorted_coef[rank - 1]
    coef = np.array([c if c >= rank_max else 0 for c in last_coef])
```

## 3.2 第二步Feature Map Reconstruction

直接利用最小平方误差，求出最终的卷积核。

```python
from sklearn import linear_model
def LinearRegression(input, output):
    clf = linear_model.LinearRegression()
    clf.fit(input, output)
    return clf.coef_, clf.intercept_
pruned_weights, pruned_bias =  LinearRegression(input=inputs, output=real_outputs)
```

## 3.3 一些细节

1. 将Relu层和卷积层分离
因为Relu一般会使用inplace操作来节省内存／显存，如果不分离开，那么得到的卷积层的输出是经过了Relu激活函数计算后的结果。

2. 每次裁减完一个卷积层后，需要对该层的bottom和top层的输入或输出大小作相应的改变。

3. 第一步求出$$ \beta $$后，若$$ \beta_i $$为0，则说明要裁减对应的channel，否则置为1，表示保留channel。


# 参考链接

[https://github.com/yihui-he/channel-pruning](https://github.com/yihui-he/channel-pruning)