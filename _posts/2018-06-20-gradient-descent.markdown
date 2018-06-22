---    
layout: post
title:  "Gradient Descent"
date:   2018-06-20 10:00:00
categories: MachineLearning
tags: MachineLearning
excerpt: 
---

## 1 梯度下降

梯度下降通过不断朝目标函数梯度负方向更新参数，从而最小化目标函数$$ J(\theta) $$的一种方式，其中$$ \theta $$是模型参数。

梯度下降的停止条件通常是（1）设置了固定的迭代轮数，达到一定轮数后退出（2）当两次迭代的目标函数之差小于某一阈值

### 1.1 几种梯度下降算法框架

#### 1.1.1 批梯度下降（Batch gradient descent）

每次对整个训练集计算参数$$ \theta $$的梯度:

$$ \theta = \theta - \eta \cdot \nabla_\theta J( \theta) $$

由于每次更新都需要计算整个训练集的梯度，因此速度较慢。如果训练集较大而内存装不下，则该方法也行不通了。

#### 1.1.2 随机梯度下降（SGD）

随机梯度下降对每条样本更新一次参数：

$$ \theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i)}; y^{(i)}) $$

SGD通常来说更快，并可以用来做online learning。

SGD的随机性也有助于找到更优的局部最优点。

#### 1.1.3 Mini-batch gradient descent

$$ \theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i:i+n)}; y^{(i:i+n)}) $$

Mini-batch方法每次从训练集中选取n个样本来更新参数。

### 1.2 Challenges

1. 选择一个合适的学习率(learning rate)很难，
过小则收敛的慢，过大则导致损失函数的值在最小值附近左右波动，甚至发散。

2. 所有特征是否采用同样的学习率

3. 对于非凸的目标函数，如何尽量避免陷入局部最小值  

4. 动态调整学习率或者设置一个阈值，社通常是事先确定好的规则，因此可能不适用所有数据集。


这里说一下模拟退火，当梯度下降到达局部最优时，依然有一定概率跳出局部最优值，
经过几次类似的移动后，也许会到达一个更优的解。若没有找到更加优秀的解，则次数达到限制后结束算法。 

首先是一个比喻，一群兔子为了找出地球上最高的山：
1. 方法一：兔子朝着比现在高的地方跳去。它找到了不远处的最高山峰。但是这座山不一定是珠穆朗玛峰。这就是爬山算法(或局部搜索法)，它不能保证局部最优值就是全局最优值。  
2. 方法二：兔子喝醉了，它随机地跳了很长时间。这期间，它可能走向高处，也可能踏入平地。但是，它渐渐清醒了并朝最高点跳去。这就是模拟退火。

模拟退火算法其实也是一种贪心算法，但是它的搜索过程引入了随机因素。模拟退火算法以一定的概率来接受一个比当前解要差的解，因此有可能会跳出这个局部的最优解，达到全局的最优解。

设$$ \delta = J(step + 1) - J(step) $$，若$$ \delta <= 0 $$，即J减小了，那么则接收这个解；
若$$ \delta > 0 $$，即下一个解比当前解要差，则以概率$$ e^{-\frac \delta T} $$接受该解，
其中T是预先设置的“初始温度”，每走一步都要乘以系数$$ r $$表示温度的下降。（之所以叫退火，也就是温度逐渐下降的过程）

### 1.2 几种梯度下降算法

#### 1.2.1 Momentum（动量）

Momentum方法可以加速SGD。SGD通常在穿过峡谷（切面在某个维度比其他维度陡的多）的时候发生振荡现象，
经常在局部极小值周围出现，SGD会沿着峡谷的斜坡缓慢的朝局部最优走。

\begin{align}
v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta) \\\
\theta &= \theta - v_t
\end{align}

其中momentum项$$ \gamma $$一般设置成0.9。
可以看出该项对于那些当前的梯度方向与上一次梯度方向相同的参数进行加强，
即这些方向上更快了；对于那些当前的梯度方向与上一次梯度方向不同的参数进行削减，
即这些方向上减慢了。因此可以获得更快的收敛速度，并在相关方向进行加速从而抑制振荡。

加上动量项就像从山顶滚下一个球，求往下滚的时候累积了前面的动量(动量不断增加)，因此速度变得越来越快。

#### 1.2.2 Nesterov accelerated gradient

从山顶往下滚的球会盲目地选择斜坡。更好的方式应该是在遇到倾斜向上之前应该减慢速度。

因此如果提前的考虑了后一步的情况，就可以阻止过快更新来提高响应性。

\begin{align}
v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta - \gamma v_{t-1} ) \\\ 
\theta &= \theta - v_t
\end{align}

通过以上两种方法，我们可以做到动态调整参数更新，从而加快SGD。
接下来看如何针对不同的参数，根据参数的重要程度来分别调整更新的幅度。

#### 1.2.3 Adagrad

Adagrad针对每个参数动态调整学习率（learning rate），对于频繁出现的特征（非稀疏的特征）进行较小的更新，
而对不频繁出现的特征（稀疏特征）。因此该方法适合处理稀疏特征的数据。

现在是针对参数向量$$ \theta $$的每个分量单独更新，设第t+1次更新时，
第i个分量为$$ \theta_i $$，则有

$$ \theta_{i,t+1} = \theta_{i,t} - \dfrac{\eta}{\sqrt{G_{t} + \epsilon}} \cdot g_{i,t} $$

其中$$ g_i = \nabla_\theta J(\theta)$$，$$ G_t = \sum_1^t g_i^2 $$


式中的参数$$ \epsilon $$是平滑项，避免出现分母为0的情况，可以取作$$ 1e-8 $$。

adagrad的其中一个好处就是学习率一般不用再人为调整了，参数$$ \eta $$一般取作0.01。

adagrad主要的缺点就是$$ G_i $$会逐渐增大，使得学习率逐渐减小，最终达到非常小的值。下面的adadelta就是解决该问题的。

#### 1.2.4 Adadelta

Adadelta是Adagrad的扩展，为了避免学习率一直单调递减。

Adagrad是所有历史的梯度的平方和，而Adadelta则是固定为前w个梯度。

第t+1次更新时，$$ E[g^2]_{t} $$的计算方法如下

$$ E[g^2]_{t} = \gamma E[g^2]_{t-1} + (1 - \gamma) g^2_{t} $$

那么参数更新的公式为

$$ \theta_{i,t+1} = \theta_{i,t} - \dfrac{\eta}{\sqrt{E[g^2]_{t} + \epsilon}} \cdot g_{i,t} $$

以上式子有个问题是更新的单位与前几种方法一样，都是$$ \Delta \theta$$与$$ \theta $$（可以假设参数单位是m，梯度即m/s，然后带入式子看一下），因此改写E为如下式子,
即不再是梯度的平方和，而是参数的平方和

$$ E[\Delta \theta^2]_{t} = \gamma E[\Delta \theta^2]_{t-1} + (1 - \gamma) \Delta \theta^2_{t} $$

记$$ {RMS[\Delta \theta]}_{t} = \sqrt {E[\Delta \theta^2]_t + \epsilon} $$，
以及$$ {RMS[g]}_{t} = \sqrt {E[g^2]_t + \epsilon} $$那么更新参数的公式如下：

$$ \Delta \theta = - \dfrac{RMS[\Delta \theta]_{t-1}}{\sqrt {RMS[g]_t}} \cdot {g_{i,t}} $$

$$ \theta_{i,t+1} = \theta_{i,t} - \Delta \theta_t $$

可以看出Adadelta不需要设置学习率了。

### Adam

在我看来就是结合了momentum以及每个参数单独更新，并且优化了起始时参数初始化为0，
而由于momentum中的$$ \gamma $$(下面公式的$$ \beta_1 $$、$$ \beta_2 $$)通常设置的较大，使得参数朝0偏倚的比较厉害的情况。

\begin{align}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\\
G_t &= \beta_2 G_{t-1} + (1 - \beta_2) g_t^2  
\end{align}

其中$$ m_t $$是第t次更新后的参数m，$$ v_t $$是第t次更新后的参数梯度平方和。

如下这样一除，也就解决了上面说的起始一段时间参数趋于0的情况：

$$ \hat m_t = \dfrac{m_t}{1 - \beta_1} $$

$$ \hat G_t = \dfrac{G_t}{1 - \beta_2} $$

那么更新参数公式为：

$$ \theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{\hat{G}_t} + \epsilon} \hat{m}_t $$

通常设置$$ \beta_1 $$为0.9，$$ \beta_2 $$为0.999，$$ \epsilon $$为1e-8，

## pserver如何更新参数

在pserver框架中通常采用mini-batch的方式来更新梯度。对于异步的控制，有三种方式：  
1. BSP（Bulk Synchronous Parallel）  
在每一轮迭代中都需要等待所有的worker计算完成，优点是每一轮迭代收敛质量高，缺点是存在慢节点的问题。
2. SSP（Stalness Synchronous Parallel）  
允许一定程度的worker进度不一致，但这个不一致有一个上限，我们称之为 staleness 值，
即最快的worker最多领先最慢的worker staleness 轮迭代。  
优点是一定程度减少了worker之间的等待时间，缺点是每一轮迭代的收敛质量不如BSP，达到同样的收敛效果可能需要更多轮的迭代。
3. ASP（Asynchronous Parallel）  
worker之间完全不用相互等待，先完成的worker，继续下一轮的训练。  
优点是速度快，缺点是适用性差，在一些情况下并不能保证收敛性。

截一张angel的图：

<img src="/images/gd/1.png" width="90%" height="90%">

我们再看一下kunpeng的实现：

图中的mServerParam和mServerGrad对应servers上的模型参数和梯度，
mWorkerParam和mWorkerGrad对应workers本地的模型参数和梯度，
mSubDatasetPtr对应当前worker的数据子集。nSync为最大延迟迭代次数，
nPull和nPush分别为从servers获取最新参数和将梯度发送给servers的频率。

通过设置nSync可以很方便地在BSP和SSP之间切换，而去除SyncBarrier就成了ASP算法的实现。

<img src="/images/gd/2.png" width="50%" height="50%">

## 参考资料

[An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html)

[https://github.com/Tencent/angel](https://github.com/Tencent/angel)

