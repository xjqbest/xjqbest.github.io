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

## 1 APPROACHES

###  1.1 Low-rank Approximation of Responses

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

我们也可以衡量一下上文的低秩假设是否有用，我们从一个CNN模型中采样了7个卷积层的输出，
并计算了对应的协方差矩阵（covariance matrix ）的特征值（eigenvalues）的从大到小之和，
称作energy，得到下图所示的结果。可以得出energy基本由最大的那些特征值贡献。
比如conv2 layer（d=256），前128个特征向量贡献了99.9%的energy。
这说明我们可以使用一小部分的卷积核来很精确的近似原始卷积核。

<img src="/images/acc_very_deep/6.jpg" width="65%" height="65%">

其实y的低秩性质，是由于卷积核W和输入x也有低秩性质，我们的优化目标直接是y的低秩子空间。


###  1.2 Nonlinear Case

前面说的方法是线性的，即没有考虑对输出做的的激活（比如ReLU）。如果我们考虑激活函数，记作$$ r $$，那么新的优化目标如下：

<img src="/images/acc_very_deep/5.png" width="40%" height="40%">

由于上式的非线性和low rank约束，我们把上式近似为：

<img src="/images/acc_very_deep/7.png" width="40%" height="40%">

其中$$ z_i $$是跟$$ y_i $$大小一样的变量，
$$ \lambda $$是正则项系数，值趋近于无穷大时，上面两个式子就一样了。

我们交替的优化$$ z_i $$和M、b：

#### 1.2.1 优化M、b
此时$$ z_i $$是固定的，当我们求出M后，b计算为$$ b = \overline z - M \overline y $$，
其中$$ \overline z $$和$$ \overline y $$是均值。将b带入得到：

<img src="/images/acc_very_deep/8.png" width="35%" height="35%">

上式跟GSVD形式一样，我们设$$ Z = z_i - \overline z $$，那么上式为：

<img src="/images/acc_very_deep/9.png" width="25%" height="25%">

这种形式的问题是“Reduced Rank Regression”，解法如下：

<img src="/images/acc_very_deep/10.png" width="50%" height="50%">

#### 1.2.2 优化$$ z_i $$

我们可以优化下式，其中$$ z_{ij} $$是向量$$ z_i $$的第j个分量，$$ y_{ij}^, $$：

<img src="/images/acc_very_deep/11.png" width="40%" height="40%">

可以用梯度下降求解。由于这里的激活函数r我们采用ReLU，
那么根据$$ z_{ij} \ge 0 $$ 还是 $$ z_{ij} < 0 $$，我们分别得到的解如下：

<img src="/images/acc_very_deep/12.png" width="33%" height="33%">

所以最终解如下：

<img src="/images/acc_very_deep/13.png" width="46%" height="46%">

### 1.3 Asymmetric Reconstruction for Multi-Layer

当我们单独的优化每一层时，浅层的错误率会快速增长，进而影响到深层。

顺便说一下，通常浅层存在的冗余比深层多，可以由下图看出：

<img src="/images/acc_very_deep/14.png" width="73%" height="73%">

我们依次近似每一层，并将上一层近似后的输出作为下一层的输入，即非对称的方法：

<img src="/images/acc_very_deep/15.png" width="40%" height="40%">

其中$$ r(Wx) = r(y) $$是这一层原始的输出（也就是该层以及前面的层都没有做近似），
$$ \hat x_i $$是上一层近似后的输出，作为该层的输入。

###  1.4 Rank Selection for Whole-Model Acceleration

通常做法是每一层都采用同样的加速比(对应的前文的$$ d' $$)
，然而这样做没有考虑到每一层冗余的情况不一样。

通过对前面提到的energy对准确率的影响，我们可以看出它们之间存在紧密的联系：

<img src="/images/acc_very_deep/16.png" width="40%" height="40%">

因此我们构造下面的目标函数：

<img src="/images/acc_very_deep/17.png" width="40%" height="40%">

其中$$ \epsilon = \prod_t \sum_{a=1}^{d'_l} \sigma_{l,a} $$
表示第l层的前$$ d'_l $$大的特征值之和。
$$ d_l $$表示第l层原始的卷积核个数，$$ C_l $$表示第l层原始的时间复杂度，
C是近似后的总时间复杂度（通过加速比计算得出）。

上式可以采用贪心法来求解，我们首先初始化每层的$$ d'_l $$为该层的$$ d_l $$值，

如果我们移除了第l层的特征值$$ \sigma_{l,d'_l} $$，那么目标函数的减小为：

\begin{align}
\Delta \epsilon / \epsilon = \sigma_{l,a} / \sum_{a=1}^{d'_l}
\end{align}

并且时间复杂度的下降为:

\begin{align}
\Delta C = \frac {1}{d_l} C_l
\end{align}

我们定义一个指标$$ \frac {\Delta \epsilon / \epsilon} {\Delta C} $$，
并去掉具有上式最小值的特征值$$ \sigma_{l,d'_l} $$，不断迭代直到时间复杂度满足要求。

这种贪心算法倾向于选择对energy减小的尽量小，并对时间复杂度减小的尽量大的那些特征值，
然后将选择的特征值去掉。

###  1.5 Higher-Dimensional Decomposition (Asymmetric 3D)

上文的方法存在的问题就是我们需要使用很小的d'来达到一定的加速，这样可能影响准确率。
为了避免d'过小，我们将上文中的方法与[其他论文](https://xujiaqi.org/2018/09/12/speed-up-cnn-with-low-rank/)结合起来，做法如下：

#### 1.5.1 确定网络结构

首先用Rank Selection得到d'，拆成两层：W'（$$ k \times k \times d' $$）
和 P($$ 1 \times 1 \times d $$ )

然后利用其他论文的方法，把W'这一层继续拆成两层：$$ k \times 1 \times d'' $$
和 $$ 1 \times k \times d' $$

经过这两种方法，我们拆成了三层：$$ k \times 1 \times d'' $$、 
$$ 1 \times k \times d' $$、$$ 1 \times 1 \times d $$。

设加速比为r，那么分摊到每种方法为$$ \sqrt r $$


#### 1.5.2 优化

前两层（$$ k \times 1 \times d'' $$ 和 $$ 1 \times k \times d' $$），
我们使用“filter reconstruction”来优化。

对于最后一层（$$ 1 \times 1 \times d $$），我们使用本文的方法，
优化后两层（$$ 1 \times k \times d' $$ 和 $$ 1 \times 1 \times d $$）。
并且采用的是非对称的方法，有助于消除累积误差。

### 1.6 Fine-tuning

Fine-tuning通常对初始模型和学习率很敏感，如果初始模型较差并且学习率很低，
那么很容易陷入一个较差的局部最优点。如果学习率太大，类似于从头训练了，
因为大的学习率导致跳出了局部最优，初始模型翔被“忘记”了一样。

本文Fine-tuning的学习率设置的比较小（1e-5），min-batch大小为128，迭代了5轮（epoch）。




