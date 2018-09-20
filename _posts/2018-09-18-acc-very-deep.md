---
layout: post
title:  "Accelerating Very Deep CNN for Classification and Detection"
date:   2018-09-18 12:00:00
categories: MachineLearning
tags: MachineLearning
excerpt: CNN
---

论文地址：
[Accelerating Very Deep Convolutional Networks for Classification and Detection](/docs/acc_very_deep/paper1.pdf)

本文提出了一种通过low-rank来重建输出的方法（response reconstruction method），
考虑了输出的非线性的情况（把输出后面接的激活函数也包括在内了，比如ReLU层）。对于非线性的情况，我们使用GSVD(Generalized Singular Value Decomposition)方法来优化。

这种显式处理非线性输出的情况，可以更好的近似非线性layer。
并且我们通过非对称的重建（其实也就是把上一层近似后的输出，作为下一层的输入，并且下一层的优化目标用了上一层原始输出，作为下层输入得到的结果）。这种非对称重建可以减小累积误差，尤其是当我们逐层近似时。

论文也提出了一种选择近似得到的矩阵的秩大小的方法（该矩阵再通过SVD分解，得到最终过的卷积核），
自适应的调整每层的压缩率。

论文在VGG16上做了实验，得到4倍加速，而top5的error只增加了0.3%

## APPROACHES

###  Low-rank Approximation of Responses

我们首先假设：输出的像素点可以用low-rank近似。

<img src="/images/acc_very_deep/1.png" width="35%" height="35%">

这个近似的网络结构也是常见：W'是d'个大小为$$ k \times k \times c $$的卷积核，
P是d个大小为$$ 1 \times 1 \times d' $$

只不过我们是要近似输出，得到一个矩阵M，然后通过对M做SVD分解得到卷积核W'和P。

具体做法如下：

我们取$$  x \in R^{k^2c+1} $$，作为原始输入reshape后的输入，
对应的一个输出$$  y \in R^d $$

原始输出是$$  y = Wx $$，其中W是大小为$$ d \times (k^2c+1) $$

我们假设y是在低秩子空间（a lowrank subspace），那么y可以写作如下形式，其中M大小为$$ d\times d $$，秩为$$ d' < d $$，y'是所有输出y的均值向量。

\begin{align}
y &= M(y - \overline y) + \overline y \\\
&= M(Wx - \overline y) + \overline y \\\
&= MWx + (\overline y - M\overline y)
\end{align}

如果我们将矩阵M做分解（比如使用SVD），即$$ M = PQ^T $$，那么上式可以写作：

\begin{align}
y = PQ^TWx + (\overline y - M\overline y)
\end{align}

我们设$$ W' = Q^TW $$ ，$$ b=\overline y - M\overline y$$，
那么新的网络结构就是前面的那个图中的b。

现在我们只要得到矩阵M，那么就可以得到新的网络结构了。可以通过优化如下式子得到M：

<img src="/images/acc_very_deep/2.png" width="35%" height="35%">

上面这个优化问题可以通过SVD或者PCA来解决。

下图是SVD方法，对矩阵M做了SVD，得到两个列正交矩阵U／V和一个对角矩阵S，然后得到P和Q：

<img src="/images/acc_very_deep/3.png" width="45%" height="45%">

下图是PCA方法，对$$ YY^T $$做了特征值分解，其中n是采样输出的个数，U是正交矩阵，S是对角矩阵，
$$ U_{d'} $$是前$$ d' $$个特征向量：

<img src="/images/acc_very_deep/4.png" width="45%" height="45%">

###  Nonlinear Case

前面说的方法是线性的，即没有考虑对输出做的的激活（比如ReLU）。如果我们考虑激活函数，记作$$ r $$，那么新的优化目标如下：

<img src="/images/acc_very_deep/5.png" width="40%" height="40%">


