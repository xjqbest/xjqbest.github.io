---
layout: post
title:  "Speeding up CNN with Low Rank Expansions"
date:   2018-09-12 12:00:00
categories: MachineLearning
tags: MachineLearning
excerpt: CNN
---


论文地址：
[Speeding up Convolutional Neural Networks with Low Rank Expansions](/docs/speed_up_cnn_with_low_rank/speed_up_with_low_rank.pdf)


下图是当输入channel个数C=1的原始卷积层，计算时间复杂度为$$ O(CNd^2H'W') $$

<img src="/images/speed_up_cnn_with_low_rank/1.png" width="35%" height="35%">

论文是基于如下事实的：

`there exists significant redundancy between different filters and feature channels`

##  Approximating Convolutional Neural Network Filter Banks

以下我们采用两种方法来近似上图的卷积：

### Schema 1

我们可以用一组更少／小的卷积核的线性组合来近似所有的卷积核。
这组卷积核称为basis filter set，记作$$ S = \{s_i\}，i \in [1, M] $$，
产生了M个输出的feature map，最终的输出是它们的线性组合，即$$ y_i = \sum_{k=1}^M a_{ik}s_k * x $$。
时间复杂度是$$ O((d^2M+MN)H'W') $$。

若有C个输入channel，则时间复杂度为$$ O(MC(d^2+N)H'W') $$。
由上式的第一项$$ CMd^2H'W' $$和第二项$$ CNMH'W' $$得出这种近似在$$ M << N $$和$$ M << d^2 $$时才会比较有效。

对于一个channel（记作c，实践中通常所有channel共用一组卷积核），
我们使用M<N个独立的卷积核$$ S^c = \{s_m^c:m\in[1,...M]\} $$，
那么每个卷积核$$ W_n^c \approx \sum_{m=1}^M a_n^{cm}s_m^c $$，
由于卷积是linear operation，卷积核重构（filter reconstruction）
和卷积操作（image convolution）可以互换，所以有如下式子：

\begin{align}
W_n * z = \sum_{c=1}^C {W_n^c * z^c} \approx \sum_{c=1}^C{\sum_{m=1}^M a_n^{cm}(s_m^c * z^c)}
\end{align}

我们也可以使用可分离（separate）卷积核来近似，即每个2D的卷积核使用两个1D的卷积核（行向量和列向量）的积来近似，
有$$ s_i * x = v_i * (h_i * x) $$，
且$$ s_i \in R^{d \times d}，v_i \in R{d\times 1}，h_i \in R^{1\times d} $$。若有C个输入channel，那么时间复杂度为$$ O(MC(d+N)H'W') $$。

<img src="/images/speed_up_cnn_with_low_rank/2.png" width="45%" height="45%">


我们在上文使用了C个独立的卷积核$$ S^1,S^2,...,S^C $$分别作用于C个输入channel，
其实在实际应用中，我们可以取$$ S^1=S^2=,...,=S^C=S $$，因为不仅不会在性能上有损失，还可以占用更小空间也更简单。


### Schema 2

方法一侧重于近似2D卷积核，每个输入channel $$ z^c $$可以看作使用M个2D的独立的卷积核近似，
探索的是输入channel的redundancy。

与方法一不同，方法二通过使用3D卷积核同时利用了输入和输出的redundancy。做法很简单：

<img src="/images/speed_up_cnn_with_low_rank/3.png" width="45%" height="45%">

每个卷积层被分解为两个卷积层，第一个卷积层有K个卷积核，每个卷积核大小是$$ d \times 1 \times C $$，
输出了K个feature map。第二层有N个卷积核，每个卷积核大小是$$ 1 \times d \times K $$。
与方法一不同的是，每一层的卷积可以同时计算。计算第一层的时间复杂度为$$ O(KCdH'W) $$，
第二层时间复杂度为$$ O(NKdH'W') $$

方法二是把公式$$ W_n * z = \sum_{c=1}^C W_n^c * z^c $$做了如下近似：

\begin{align}
W_n * z \approx h_n * V = \sum_{k=1}^K {h_n^k * V^k} = \sum_{k=1}^K{h_n^k * (v_k * z)} 
= \sum_{k=1}^K{h_n^k * \sum_{c=1}^C{v_k^c * z^c }} = \sum_{c=1}^C[\sum_{k=1}^K{h_n^k * v_k^c}] * z^c
\end{align}

可以看出上式符合可分离卷积核的形式，所以也可以说第二种方法就是近似得到可分离卷积核。

## Optimization

如何得到最优的一组基卷积核呢，我们有两种优化办法，第一种是最小化重建filter的误差，第二种是最小重建输出的误差。

### Filter Reconstruction Optimization

#### 针对方法一

<img src="/images/speed_up_cnn_with_low_rank/4.png" width="44%" height="44%">

其中第二项是核范数（nuclear norm），是指矩阵奇异值的和。加上第二项的目的是为了让这M个卷积核尽量相互独立。

我们通过不断交替的优化$$ s_m $$和$$ a_n $$来得到上式的最小值。

#### 针对方法二

<img src="/images/speed_up_cnn_with_low_rank/5.png" width="34%" height="34%">

我们通过直接利用可分离卷积的形式避免了优化方法一中的nuclear norm。
利用共轭梯度法（conjugate gradient descent），交替优化上式中的行向量和列向量。

### Data Reconstruction Optimization

我们可以通过重建输出得到最优的一组基卷积核：

<img src="/images/speed_up_cnn_with_low_rank/6.png" width="42%" height="42%">

方法二有两个优点：
1. 卷积核中不相关或者冗余的维度，在重建输出时候会被自动的忽略，
而方法一中，这些维度还是会被考虑进去。
2. 通过逐层的优化，我们可以把上一个卷积层优化后的输出，喂给下一个卷积层，
而不是把原始的输出喂给下一个卷积层。这样做可以在优化当前层时，也考虑了前一层输出的误差。

<img src="/images/speed_up_cnn_with_low_rank/7.png" width="60%" height="60%">

一个明显的可选方法是直接基于新的近似的网络结构，然后重新训练得到模型。
不过实际效果并不好，容易产生过拟合，而使用dropout又会产生欠拟合，
这很可能是由于近似的网络结构已经能充分近似原始卷积核了。

## 实验

对于filter reconstruction optimization，我们优化基卷积核直到误差足够小。

对于data reconstruction optimization，我们逐层优化，并且可以joint optimization（也就是说把上一层的近似后的输出，喂给下一层，而不是上一层原始的输出）。

实验采用四层的CNN模型如下：

<img src="/images/speed_up_cnn_with_low_rank/8.png" width="65%" height="65%">

第一层卷积很难／无法近似，因为它直接作用于输入的原始像素。而第四层的卷积核大小是$$ 1 \times 1 $$，
我们也不对它做近似了。从图中也可以看出，第二层和第三层的计算时间占到了90%，因此我们只近似这两层。


下图是实验结果，可以看出joint data optimization效果最好。
方法二可以达到2.5倍加速而不损失精度，4.5倍加速只损失了1%精度。

<img src="/images/speed_up_cnn_with_low_rank/9.png" width="75%" height="75%">