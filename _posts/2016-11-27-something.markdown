---
layout: post
title:  "机器学习一些概念"
date:   2016-11-27 15:34:01
categories: MachineLearning
tags: MachineLearning
excerpt: 记录自己学习中遇到的机器学习的概念，不定期更新，随时记录
---

# 一些概念

## 最大后验估计和最大似然估计

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

## cross-entropy error function
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

## 其他

1. 正确分类与训练集不同的新样本能力叫做泛化。  
bootstrap：有放回的随机抽样。  
在最大似然中，似然函数的负对数叫做误差函数（error function）。

2. 类似多项式函数的这种关于未知参数满足线性关系的函数，叫做线性模型  。
例如： $$ y = w_0 + w_1x + w_2x^2 + ... + w_Mx^M $$  
有着更大M值的更复杂（灵活）的多项式被过分的调参，使得多项式被调节成了与目标值的随机噪声相符。

3. 对于一个给定的模型复杂度，当数据集的规模增加时，过拟合问题变得不那么严重了。即数据集的规模越大，我们能够用来拟合数据的模型就可以越复杂。  
通常用来控制过拟合的一种技术是正则化，一般$$ w_0 $$从正则化项中省略。

4. 监督学习分为生成方法和判别方法。  
生成方法：由数据学习联合概率分布$$ P(X,Y) $$，然后求出条件概率分布$$ P(Y|X) $$作为预测的模型。例如：朴素贝叶斯、隐马尔可夫。  
判别方法：由数据直接学习决策函数$$ f(X) $$或者条件概率分布$$ P(Y|X) $$作为预测的模型。例如：k邻近、决策树、LR、SVM等。

5. 最大似然方法低估了分布的方差，即偏移(bias)的现象。

6. 曲线拟合问题  
曲线拟合目标是根据N个输入**x**= $$ (x_1,x_2,...,x_N)^T $$ ，和其对应的目标值 **t** = $$ (t_1,t_2,...,t_N)^T $$ 。 在给定一个x新值时候，对目标变量t进行预测。  
我们可以用概率来表达目标变量t的不确定性，假定给定x的值，对应的t值服从高斯分布，设分布的均值为$$ y(x, \mathbf w) $$。
其中 $$ y(x, \mathbf w) = w_0 + w_1x + w_2x^2 + ... + w_Mx^M $$。设 $$ \beta $$ 为高斯分布的方差的倒数，那么有：
\begin{align}
p(t|x,\mathbf w,\beta)=N(t|y(x,\mathbf w,\beta^{-1}))
\end{align}
我们用训练数据$$ \{\mathbf x,\mathbf t\} $$，用最大似然法，来确定参数$$ \mathbf w $$和$$ \mathbf \beta $$的值。似然函数为：
\begin{align}
p(\mathbf t|\mathbf x,\mathbf w,\beta) = \prod_{n=1}^N{N(t_n|y(x_n,\mathbf w, \beta^{-1}))}
\end{align}
可以得到对数似然函数为：
\begin{align}
\ln p(\mathbf t|\mathbf x,\mathbf w, \beta) = -\frac{\beta}{2}\sum_{n=1}^N{\\\{y(x_n,\mathbf w) - t_n\\\}} + \frac{N}{2}\ln{\beta} - \frac{N}{2}\ln(2\pi)
\end{align}
对于确定$$ \mathbf w $$的问题来说，最大化似然函数等价于最小化平方和误差函数,设最大似然解为$$ {\mathbf w}_{ML} $$.同样的使用最大似然方法确定
$$ \beta $$，有：
\begin{align}
\frac{1}{\beta_{ML}}=\frac{1}{N}\sum_{n=1}^N{\\\{ y(x_n, \mathbf w_{ML} ) - t_n \\\}}
\end{align}
现在让我们朝着贝叶斯方法前进一步，引入在多项式系数$$ \mathbf w $$上的先验分布。并设其服从高斯分布，则有：
\begin{align}
p(\mathbf w, \alpha)=N(\mathbf w|\mathbf 0, \alpha^{-1}\mathbf I)= (\frac{\alpha}{2\pi})^{\frac {M+1} 2} exp \\\{ -\frac{\alpha}{2}{\mathbf w}^T \mathbf w\\\}
\end{align}
其中$$ \alpha $$为超参数（控制模型参数分布的参数）。使用贝叶斯定义，$$ \mathbf w $$的后验概率正比于先验分布和似然函数的乘积，即：
\begin{align}
p(\mathbf w|\mathbf x,\mathbf t, \alpha, \beta) \propto p(\mathbf t| \mathbf x, \mathbf w, \beta) p(\mathbf w|\alpha)
\end{align}
给定数据集，我们现在通过寻找最可能的$$ \mathbf w $$值来确定$$ \mathbf w $$，即最大后验。取上式的负对数，可以看到最大化后验概率就是最小化下式，
最大化后验概率等价于最小化正则化的平方和误差函数（其中正则化参数为$$ \frac \alpha \beta $$）：
\begin{align}
\frac{\beta}{2}\sum_{n=1}^N{\\\{y(x_n,\mathbf w) - t_n\\\}} + \frac{\alpha}{2}{\mathbf w}^T \mathbf w
\end{align}
贝叶斯方法就是自始至终的使用概率的加和和乘积规则，，因此预测概率可以写成如下形式（省略了$$ \alpha $$和$$ \beta $$）：
\begin{align}
p(x|t,\mathbf x,\mathbf t) = \int{p(t|x,\mathbf w)p(\mathbf w|\mathbf x, \mathbf t)d\mathbf w}
\end{align}

7. 维度灾难，是指数据维数很高时，很多机器学习问题变的困难。在高维空间中参数配置数目远大于样本数目。
比如把输入空间分割成一个个的单元格，对于需要区分的d维以及v个值，我们需要$$ O(v^d) $$个区域和样本。  
在高维空间中，一个球体的大部分体积都聚集在表面附近的薄球壳上。
比如高斯分布的概率质量集中在薄球壳上。

8. 最小化错误分类率是指尽可能少的做出错误分类。我们将输入空间划分为不同的区域$$ R_k $$，即决策区域，区域间的边界叫做决策边界。
每个类别都有一个决策区域，区域$$ R_k $$中的所有点都被分到$$ C_k $$类。每一个决策区域未必是连续的。  
把每个x分配到后验概率$$ p(C_k|x) $$最大的类别中，那么我们的分类错误的概率就会最小。

9. 最小化期望损失，可以通过损失函数（loss function）来描述，即是对于所有可能的决策或者动作可能产生的损失的一种整体的度量。
假设对于新的x值，真实的类别为$$ C_k $$，我们把x分类为$$ C_j $$时可能造成某种程度的损失，记做$$ L_{kj} $$，即损失矩阵（loss matrix）。  
最优解是使损失函数最小的解，但是这依赖于真实的类别，它是未知的。对于一个输入向量$$ \mathbf x $$，
对于真实类别的不确定性通过联合概率分布$$ p(\mathbf x, C_k) $$表示，因此我们转而去最小化平均损失，为：
\begin{align}
E[L] = \sum_k{\sum_j{\int_{R_j}{L_{kj}{}p(\mathbf x, C_k)d \mathbf x}}}
\end{align}
对于每个新的$$ \mathbf x $$，把它分到能使下式取得最小值的第j类：  
\begin{align}
\sum_k{L_{kj}p(C_k|\mathbf x)}
\end{align}

10. 推断和决策  
（1）使用贝叶斯定理确定后验概率。或者直接对联合概率分布$$ p(\mathbf x, C_k) $$建模，然后归一化得到后验概率。
得到后验概率后再决定每个输入$$ \mathbf x $$的类别。显示或者隐式对输入及输出建模的方法被称为生成式模型。  
（2）首先确定后验概率$$ p(C_k|\mathbf x) $$，，接下来对输入$$ \mathbf x $$进行分类。这种直接对后验概率建模的方法被称为判别式模型。  
（3）找到一个函数$$ f(\mathbf x) $$，被称为判别函数，将输入直接映射为类标签。

11. 信息论  
随机变量x的熵：$$ H(x)=-\sum_x{p(x)\log_2{p(x)}} $$。当遇到一个x使得p(x)=0，令$$ p(x)\log_2{p(x)} = 0$$.  
熵是传输一个随机变量状态值所需的比特位的下界。  
对于多元连续变量上的概率密度，微分熵为：
\begin{align}
H[\mathbf x] = -\int{p(\mathbf x)\ln{p(\mathbf x)}d \mathbf x}
\end{align}
设有联合概率分布$$ p(\mathbf x, \mathbf y) $$，我们从概率分布中抽取了一对$$ \mathbf x $$和$$ \mathbf y $$。
如果$$ \mathbf x $$的值已知，那么需要确定对应的$$ \mathbf y $$值需要的附加信息是$$ -\ln{p(\mathbf y|\mathbf x)} $$，
因此用来确定$$ \mathbf y $$值的平均附加信息为：
\begin{align}
H[\mathbf y|\mathbf x] = -\int{\int{p(\mathbf x, \mathbf y)\ln(p(\mathbf y|\mathbf x))}d \mathbf y d \mathbf x}
\end{align}
可以看出条件熵满足：
\begin{align}
H[\mathbf y , \mathbf x] = H[\mathbf y|\mathbf x] + H[\mathbf x]
\end{align}
考虑一个未知的分布$$ p(\mathbf x) $$，使用一个近似的分布$$ q(\mathbf x) $$对它建模。如果使用$$ q(\mathbf x) $$建立一个编码体系，
用来把$$ \mathbf x $$的值传给接收者，由于使用的$$ p(\mathbf x) $$，而不是真实的$$ p(\mathbf x) $$，于是我们需要的附加信息。平均附加信息为：
\begin{align}
KL(p||q) &= -\int{p(\mathbf x)ln(q(\mathbf x))d \mathbf x} - (-\int{p(\mathbf x)ln(p(\mathbf x))d \mathbf x}) \\\
&= -\int{p(\mathbf x)ln{\\\{ \frac{q(\mathbf x)}{p(\mathbf x)} \\\}d \mathbf x}}
\end{align}
即KL散度，可以看作两个分布$$ p(\mathbf x) $$和$$ q(\mathbf x) $$之间的不相似程度的度量。  
可以通过考察联合概率分布与边缘概率分布乘积之间的KL散度来判断他们是否接近于相互独立，即互信息：
\begin{align}
I[\mathbf x,\mathbf y] &= KL(p(\mathbf x,\mathbf y)||p(\mathbf x)p(\mathbf y)) \\\
&= - \int{\int{p(\mathbf x,\mathbf y)\ln(\frac{p(\mathbf x)p(\mathbf y)}{p(\mathbf x,\mathbf y)})}d \mathbf x d \mathbf y}
\end{align}
互信息与条件熵的关系为：
\begin{align}
I(\mathbf x,\mathbf y) = H[\mathbf x] - H[\mathbf x|\mathbf y] =  H[\mathbf y] - H[\mathbf y|\mathbf x]
\end{align}
即表示一个新的观测$$ \mathbf y $$造成的$$ \mathbf x $$的不确定性的减小。


## 参考资料
[最大似然估计（MLE）和最大后验概率（MAP）](http://blog.csdn.net/upon_the_yun/article/details/8915283)  
[最大后验估计和最大似然估计](http://www.cnblogs.com/emituofo/archive/2011/12/02/2271410.html)  
[Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)