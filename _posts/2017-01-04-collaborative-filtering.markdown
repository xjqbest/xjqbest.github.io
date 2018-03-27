---
layout: post
title:  "Collaborative Filtering"
date:   2017-01-04 23:28:00
categories: MachineLearning
tags: MachineLearning
excerpt: Collaborative Filtering
---

# 常见推荐系统算法

## 基于人口统计学的推荐

只是简单的根据系统用户的基本信息发现用户的相关程度，然后将相似用户喜爱的其他物品推荐给当前用户。

![](/images/cf/1.jpg)

系统首先会根据用户的属性建模，比如用户的年龄，性别，兴趣等信息。根据这些特征计算用户间的相似度。
比如系统通过计算发现用户A和C比较相似。就会把A喜欢的物品推荐给C。

优势：

1. 不需要历史数据，没有冷启动问题

2. 不依赖于物品的属性，因此其他领域的问题都可无缝接入。

不足：

1. 算法比较粗糙，效果很难令人满意，只适合简单的推荐

## 基于内容的推荐

与上面的方法相类似，只不过这次的中心转到了物品本身。使用物品本身的相似度而不是用户的相似度。

![](/images/cf/2.jpg)

系统首先对物品（图中举电影的例子）的属性进行建模，图中用类型作为属性。在实际应用中，只根据类型
显然过于粗糙，还需要考虑演员，导演等更多信息。通过相似度计算，发现电影A和C相似度较高，因为他们都
属于爱情类。系统还会发现用户A喜欢电影A，由此得出结论，用户A很可能对电影C也感兴趣。于是将电影C推荐给A。

优势：

1. 对用户兴趣可以很好的建模，并通过对物品属性维度的增加，获得更好的推荐精度

不足：

1. 物品的属性有限，很难有效的得到更多数据
2. 物品相似度的衡量标准只考虑到了物品本身，有一定的片面性
3. 需要用户的物品的历史数据，有冷启动的问题

## 协同过滤
协同过滤是推荐算法中最经典最常用的。

协同过滤是利用集体智慧的一个典型方法，协同过滤一般是在海量的用户中发掘出一小部分和你品位比较类似的，
在协同过滤中，这些用户成为邻居，然后根据他们喜欢的其他东西组织成一个排序的目录作为推荐给你；
或者在海量的物品中发掘出相似的物品，然后根据你喜欢的物品，推荐相似的物品给你。

即有下列问题：

1. 如何确定一个用户是不是和你有相似的品位？

2. 如何确定一个物品是不是和你所喜欢的物品相似？

## Collaborative Filtering（协同过滤）

要实现协同过滤，需要一下几个步骤：

1. 收集用户偏好

2. 找到相似的用户或物品

3. 计算推荐

### 收集用户偏好

要从用户的行为和偏好中发现规律，并基于此给予推荐，如何收集用户的偏好信息成为系统推荐效果最基础的决定因素。例如：

![](/images/cf/3.png){: width="600px" height="300px"} 

收集了用户行为数据，我们还需要对数据进行一定的预处理，其中最核心的工作就是：减噪和归一化。

进行的预处理后，根据不同应用的行为分析方法，可以选择分组或者加权处理，之后我们可以得到一个用户偏好的二维矩阵，一维是用户列表，
另一维是物品列表，值是用户对物品的偏好，一般是 [0,1] 或者 [-1, 1] 的浮点数值。

### 找到相似的用户或物品

当已经对用户行为进行分析得到用户喜好后，就可以根据用户喜好计算相似用户和物品，然后基于相似用户或者物品进行推荐，
这就是最典型的协同过滤的两个分支：基于用户的协同过滤（UserCF）和基于物品的协同过滤（ItemCF），这两种方法都需要计算相似度。

#### 相似度的计算

现有的几种基本方法都是基于向量的，其实也就是计算两个向量的距离，距离越近相似度越大。在推荐的场景中，
在用户-物品偏好的二维矩阵中，我们可以将一个用户对所有物品的偏好作为一个向量来计算用户之间的相似度，或者将所有用户
对某个物品的偏好作为一个向量来计算物品之间的相似度。

常见的方法：

1. Euclidean Distance  
\begin{align}
d(\vec i,\vec j) = ||\vec i - \vec j||
\end{align}
则有：  
\begin{align}
sim(\vec i,\vec j) = \frac {1}{1 + d(\vec i,\vec j)}
\end{align}

2. Cosine Similarity  
\begin{align}
sim(\vec i,\vec j) = cos(\vec i, \vec j) = \frac {\vec i \cdot \vec j}{||\vec i|| \times ||\vec j||}
\end{align}

3. Tanimoto Coefficient  
\begin{align}
sim(\vec i,\vec j) = \frac {I \cap J}{I \cup J}
\end{align}

4. Pearson Correlation Coefficient  
\begin{align}
sim(\vec X,\vec Y) = \frac {\sum (X_i - \overline X)(Y_i - \overline Y)}{\sqrt {\sum (X_i - \overline X)^2} \sqrt {\sum (Y_i - \overline Y)^2}}
\end{align}

#### 预测

##### ItemCF

设$$ N(u) $$是用户喜欢的物品集合，$$ S(i,k) $$是与物品i相似的k个物品的集合，$$ s_{ij} $$是物品i与物品j的相似度，
$$ r_{ui} $$是用户u对物品i的评分，$$ P_{ui} $$是用户u对物品i的评分，那么有：  

\begin{align}
p_{ui} = \frac {\mathop{\sum }\limits_{j \in N(u) \cap S(i,k)} s_{ij} r_{uj}} {\mathop{\sum }\limits_{j \in N(u) \cap S(i,k)} |s_{ij}| }
\end{align}

## ItemCF 的分布式实现

用户输入格式：每行为一个user_id及其所有item，其中item是<item_id, rating>对的形式。

计算相似度使用Cosine Similarity.

看下图的一些观察：

1. 如果两个item没有共同用户的评分的话，那么Cosine Similarity为零（例如，item2和item3，item5和item6等）  


2. 矩阵是稀疏的

![](/images/cf/4.png){: width="500px" height="250px"} 

### 算法流程

假设我们使用map-reduce框架，它还有着参数服务器可以存key-value对，称之为table。

下面说的是计算相似度的过程，直接计算的话大量的时间都浪费在了无用的求出来的为0的向量分量上，
而下面的算法中map的过程中就把为零的向量分量直接去掉了，不参与计算。

#### 计算向量的点积：

1. 用map把输入流转成一个个的< <item_id1, item_id2>, <rating1, rating2> >对。

2. 写入table过程中，执行aggregrate，即对于key相同的value，先执行$$ key \times value $$，再与之前计算的部分点积加起来，
最终写入table中的即为求出的点积。

其中，map的过程是：

```cpp
itemi_itemj_pairs; (<<string, string>, <float, float>>, that is <<item_id1, item_id2>, <rating1, rating2>>)
for user in all users:
	all items whose rating is not zero are made pairs with each other.(can keep half of them, others are redundent)
	add these pairs to itemi_itemj_pairs
```


#### 计算向量的模

1. 使用map把输入数据转换成<item_id, rating>

2. 写入table，把相同key的value加起来，然后求sqrt。

## 参考资料

[基于领域的协同过滤算法 ： UserCF and ItemCF](https://www.zybuluo.com/xtccc/note/200979)  
[深入推荐引擎相关算法](https://www.ibm.com/developerworks/cn/web/1103_zhaoct_recommstudy2/)  
 