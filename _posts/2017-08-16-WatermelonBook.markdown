---
layout: post
title:  "西瓜书小笔记"
date:   2017-08-16 22:14:01
categories: MachineLearning
tags: MachineLearning
excerpt: 西瓜书小笔记
---

## 第三章 线性模型

（1）linear model

线性模型： $$ f(\mathbf x) = {\mathbf w}^T x + b $$ ， $$ \mathbf w $$和b学到后，模型就确定了。

线性回归：

对于离散属性（列名属性），如果属性之间存在序关系，可以通过连续化将其转化为连续值（数值属性）。
比如三值属性“高”、“中”、“低”，可以转化为1.0、0.5、0.0（1与0.5比较接近，与0比较远）。

为了衡量f(x)与y的差别，可以采用均方误差（对应欧氏距离），让均方误差最小化。
基于均方误差最小化来进行模型求解的方法称为“最小二乘法”。在线性回归中，
最小二乘法就是试图找到一条直线，使得所有样本到直线上的欧氏距离之和最小。

一般地，考虑单调可微函数g，令
\begin{align}
y = g^{-1}({\mathbf w}^T x + b)
\end{align}
这样的模型叫做广义线性模型。例如$$ g $$可以是$$ ln $$

logistic regression:

对数几率函数（此处是sigmoid函数）：
\begin{align}
y = \frac{1}{1 + e^{-( {\mathbf w}^T x + b )} }
\end{align}

对于二分类：
\begin{align}
p(y=1|\mathbf x) = \frac{e^{ {\mathbf w}^T x + b } }{1 + e^{ {\mathbf w}^T x + b } } \\\
p(y=0|\mathbf x) = \frac{ 1 }{1 + e^{ {\mathbf w}^T x + b } }
\end{align}

可以通过极大似然法来估计$$ \mathbf w $$和b，最大化对数似然，有：
\begin{align}
l(\mathbf w, b) = \sum_{i=1}^{m}{ln p(y_i|{\mathbf x}_i;\mathbf w, b)}
\end{align}
其中：
\begin{align}
p(y_i|{\mathbf x}_i;\mathbf w, b) = y_i p(y=1|{\mathbf x}_i;\mathbf w, b) + (1-y_i) p(y=0|{\mathbf x}_i;\mathbf w, b)
\end{align}

（2）线性判别分析(LDA)

LDA的思想：给定一个训练集，设法将样本投影到一条直线上，使得同类样本的投影点尽可能接近，异类样本的投影点尽可能远离。
在对新样本进行分类时，将其投影到同样的这条直线上，再根据投影点的位置来确定新样本的类别。它是有监督的降维技术。

对于二分类，设$$ \mu_i $$表示第i类的均值向量，$$ \sum_i $$表示第i类的协方差矩阵。
欲使同类样本的投影点尽可能接近，可以让同类样本投影点的协方差尽可能小，即$$ w^T\sum_0{}w + w^T\sum_1{}w$$尽可能小。
欲使异类样本的投影点尽可能远离，可以让类中心点的距离尽可能大，即$$ ||w^T\mu_0 - w^T\mu_1||^2_2 $$尽可能大。