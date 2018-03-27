---
layout: post
title:  "不均衡分类问题"
date:   2016-12-1 20:18:02
categories: MachineLearning
tags: MachineLearning
excerpt: 不均衡分类问题总结
---

# Introduction
不均衡分类问题，是指当某个类别下对应的样本点的数量远小于其它类别时的分类问题。  
这一问题多产生于类似于异常检测的场景：大部分样本都是正常的负样本，只有极少数的样本为正样本。  

设训练集$$ S $$有$$ m $$个样本（即$$ |S|=m $$）。其中$$ S=\{(x_i,y_i)\}，i=1,...,m ，x_i \in X$$是
n维特征空间X($$ X=\{f_1,f_2,...,f_n\} $$)中的一个向量，$$ y_i \in Y=\{1,...,C\} $$是相应的类标签。
当$$ C=2 $$表示二分类问题，定义少数类(minority class)为$$ S_{min} \subset S $$，多数类(majority class)为
$$ S_{maj} \subset S $$，并且满足$$ S_{min} \cap S_{maj}=\{\emptyset\} $$和$$ S_{min} \cup S_{maj}=\{S\} $$.  

本文以二分类为例子，负样本为多数类，正样本为少数类。

# Solutions for Imbalanced Learning

## Sampling Methods
采样方法对数据集进行采样，从而使各个类别的数量接近。这类方法主要可以分为针对负样本的
欠采样（选择部分点参与分类）与针对正样本的过采样（产生新的点参与分类）。

#### Undersampling(欠采样)
下采样的目标是使个各个类别下的点的数量基本一致。负样本中的点可以分为四类：
 
1. Noise: 即样本点标签错误的点，如图1左侧左下角的点。
2. borderline: 即靠近分界面的一部分点，这部分点容易被分类错误，或者导致正样本分类错误，可以看做另一类噪音点。
3. redundant: 冗余的点，这部分点不影响分类结果。
4. safe: 即能使用分类结果正确必须保留点，直观上，包括一些靠近分类面但不属于borderline类的点。

在理想的情况下，我们仅保留safe的点，从而避免受到noise与borderline中的点的影响，且使分类面与真实分类面一致。
在下采样的各种方法中，大都利用到了类似的思想，对不同的点进行不同的处理。  
![](/images/imbalanced_learning/1.png){: width="400px" height="180px"}  
图一，包括了noise, borderline, redundant与safe四类点  
![](/images/imbalanced_learning/2.png){: width="400px" height="180px"}  
图二，去除了noise与borderline的点。可以直观看出borderline点需要过拟合才可能分类正确  
![](/images/imbalanced_learning/3.png){: width="400px" height="180px"}  
图三，去除了redundant的点

#### Oversampling(过采样)
根据现有的正样本，在数据集中通过一定方法生成更多的正样本，从而使不同类别的样本点数量接近。  
相对于下采样，上采样的方法不会丢失数据集中重要的信息。

***

### (1) Random Oversampling / Undersampling
随机过采样缺点：多个同一样本的副本容造成过拟合，而且使得分类规则具体化，泛化性能不好。  
随机欠采样缺点：移除了多数类的一些样本造成信息的缺失，可能导致对于多数类的欠拟合。

### (2) Informed Undersampling

#### EasyEnsemble
主要为了解决随机欠采样的信息损失问题  
使用了集成学习，流程为：

1. 从多数类中独立取出若干子集
2. 将每个子集与少数类的数据合起来训练生成多个基分类器(使用AdaBoost得到每个弱分类器)
3. 将这些基分类器组合形成一个集成分类器，各个分类器的权值一致

![](/images/imbalanced_learning/4.png){: width="380px" height="280px"}  

#### BalanceCascade
主要为了解决随机欠采样的信息损失问题  
在每一次得到一个基分类器后，将此基分类器可以正确分类的点从$$ S_{maj} $$中去除。  
去掉了$$ S_{maj} $$的冗余信息，尽可能用到更多有用的信息。
 
![](/images/imbalanced_learning/5.png){: width="420px" height="390px"}   

#### KNN Undersampling
基于knn的四种欠采样方法：

1. NearMiss-1  
选择距离最近三个少数类样本平均距离最小的那些多数类样本
2. NearMiss-2  
选择距离最远三个少数类样本平均距离最小的那些多数类样本
3. NearMiss-3  
对于每一个少数类样本，选择一定数量的最近多数样本，来保证每个少数类都被一些多数类样本包围
4. most distant  
选择距离最近三个少数类样本平均距离最大的那些多数类样本

可以看出，NearMiss-1，2，3相当于主要是去除redundant的点，并且期望的结果是：nearmiss方法比random和distant效果更好，random比distant效果更好，
nearmiss-3是高precision低recall，distanct是高recall低precision.

### (3) Synthetic Sampling

#### SMOTE(Synthetic Minority Oversampling Technique)
SMOTE算法是利用特征空间中的少数类样本之间的相似性来生成人工数据。  
算法如下：

```
输入: T, 原始数据集; n, 对每个原样本点产生多少个新样本  
输出: S, 新产生的样本点
    train a k-NN with all the positive points(minority class) 
    for(each point p in positive class){
        find the k nearest neighbors of p with k-NN
        for(i in 0..n){
            choose a random point q from the k nearest neighbors;
            s = p + rand[0, 1] * (q - p);
            S += s;
        }
    }
    return S
```

可以看出 SMOTE 算法是建立在相距较近的少数类样本之间的样本仍然是少数类的假设基础上的。

#### Borderline-SMOTE
在最基本的SMOTE算法中，并未对正例中的样本点进行区分，而是随机选取。这样，如果选择的
两个正样本均比较靠近正样本区域的内部，则产生的新的样本点将很可能属于redundant的点。  
针对这一问题，bSMOTE根据k近邻中负样本点的数量m，将正样本点分为三类：  
1. m == k: 说明该正样本点周围全是负样本点，该正样本很可能是噪音，后续不做处理  
2. k/2 <= m < k : 半数或以上的点是负样本点，该正样本被加入到 DANGER集中  
3. 0 <= m < k/2: 该样本点属于安全的样本点，即在正样本区域内部

然后，bSMOTE-1只针对DANGER集中的正样本点执行原始的SMOTE过程。一个bSMOTE-1的例子如下图：

![](/images/imbalanced_learning/6.png){: width="600px" height="200px"}  
(a)原始数据集 (b)从原始正样本中选出DANGER的样本点并标蓝 (c) 产生了新数据点之后的数据集

bSMOTE-2与bSMOTE-1类似，不同的地方在于，在对DANGER集中的顶点执行SMOTE过程时，对每一个DANGER集中的点，
也会从其k个最近邻中的负样本中选择点来生成新的样本点，只是此时生成的样本点时，随机数选择范围为[0, 0.5], 
从而保证生成的点位于正样本的区域内。

#### ADASYN(Adaptive Synthetic Sampling)
前述方法中，对每一个执行SMOTE的候选点，所生成的样本数量是一致的，从而削弱了边界点的作用。
ADASYN针对这一问题，根据正样本点靠近边界的程度为正样本点设定不同的权值，权值越高，生成的样本点越多。  
具体的做法为：  
设$$ m_s $$和$$ m_l $$分别为少数类样本数和多数类样本数（则有$$ m_s \le m_l $$和$$ m_s + m_l = m $$）  
1. 计算需要生成的样本数$$ G=(m_l-m_s)\times \beta $$，其中参数$$ \beta \in [0,1] $$  
2. 对每个$$ x_i \in minorityclass $$，找到其k临近(跟原始的SMOTE不同，原始的SMOTE只在少数类类中找)  
计算权值$$ r_i = \Delta_{i}/K $$，其中$$ i=1,...,m_s ，\Delta_{i} $$是$$ x_i $$的K临近中多数样本数量，则有$$ r_i \in [0,1] $$  
3. 归一化： $$ \hat r_i = r_i / \sum_{i=1}^{m_s}r_i $$  
归一化后满足：$$ \sum_i{\hat r_i} = 1 $$  
4. 得到每个点$$ x_i $$需要生成的新样本数$$ g_i = \hat r_i \times G $$

### (4) Sampling with Data Cleaning Techniques

#### CNN(Condensed Nearest Neighbor)
该方法的目标是选择整个训练集（包括正负样本）的一个子集，在该子集上使用kNN分类器，
训练集中所有的样本点的分类结果与原始数据集一致。针对这一目标，作者提出了一个迭代的算法:

```
输入: T, all the points
     k, k for the kNN classifier
输出: S, T的子集且训练数据集的分类结果不变
算法:
    Move some initial points to S //对于采样来说，一般是把所有正样本和一个随机的负样本加入
    for(every point p in T){
        all_right = true
        if(S can not classify p rightly with kNN){
            Add p to S;
            all_right = false;
        }
        if(all_right){
            break //如果某轮循环中所有点都分类正确，则找到相应一致子集
        }
    }
    return S
```
在Undersampling的场景下，该一致子集被作为Undersampling的结果并参与后续分类。

#### Tomek links + CNN
记b = nno(a) 为对于点a, 离其最近且不属于同一类的点。  
对于点x, y, 如果x, y属于不同的类别，且y = nno(x), x = nno(y), 则点对<x, y>构成Tomek Link

如果一对点构成Tomek Link, 则其中有一个点为噪音，或两个点均为borderline点。
因此，在这种情况下，需要去除点x或点y. 从原始数据集中打破 Tomek Link后，在所得到的数据集上进行CNN，
仍然可以确保得到一致子集。因此，从Undersampling的角度来说，可以去除属于Tomek Link中的负样本点，
在得到的数据集上进一步执行CNN。

#### Modified CNN
在原始的CNN方法中，并没有对负例中的样本点进行区分。这样可以导致两个问题：  
1. 存在部分redundant的点被加入到一致子集中，从而导致后续分类的额外开销。  
2. 并非所有的safe的点被加入到一致子集中，而是被另一些redundant的点取代

为了解决以上问题，作者提出了改进的策略：  
方法的思想是选择靠近真实分类面的点。但是由于真实分类面是不可知的，所以需要进行估计并选出边界点。  
记b = nno(a) 为对于点a, 离其最近且不属于同一类的点。

如下图所示，y = nno(x), 则可以认为y是靠近分类面的点，
这些类似于y的点，可以做为一致子集中的点的候选。但是，这一规则不足以加入所有的点。如下图所示，对于集合A中的
点，它们也是靠近分类面的点，并且对于正确分类是必须的。但是使用以上规则，这些点不可能被加入到候选子集中。
所以，对于一个点，比如z, 那么u = nno(z),  我们从所有能够正确分类z的点（即dist(z, *) < dist(z, u)）中选
出u的最近邻v, 则v也是靠近分类边界的点并且对于正确分类 z是必须的，因此v需要被加入到一致子集中。

![](/images/imbalanced_learning/7.png){: width="460px" height="170px"} 

```
输入: T, all the points
输出: S, T的子集且训练数据集的分类结果不变
算法:
    Move some initial points to S //对于采样来说，一般是把所有正样本和一个随机的负样本加入
    F = {} //F是边界点的集合，作为一致子集S的候选，所有加入到S中的点均来自F（规则一）或类似于规则二中的v
    for (every point p in T){
        all_right = true
        if (S can not classify p rightly with 1-NN){
            if(p in F){ //p is already a frontier point, add p to S
                S += p
                F -= p
            }
            else if (F can classify p with 1-NN) {              
                S += p
                F -= p 
            }      
            else { //apply rule 1 and rule 2
                find q = nno(p)
                add q to F
                     
                find t = nno(q) from A, where A = {w|dist(w, p) < dist(q, p)}
                add t to S         
            }
            all_right = false
        }
        if (all_right){
            break //如果某轮循环中所有点都分类正确，则找到相应一致子集
        }
    }
    return S
```

#### OSS (One Side Selection)
基于最开始对点的分类，OSS首先执行基于1-NN的CNN算法，然后对结果集，
去除所有Tomek Link点对中的负样本。第一步CNN的作用是去除数据集中的Redundant的点，
而第二步的目的是去除数据集中的Noise与Borderline点。

#### ENN (Edited Nearest Neighbor)
ENN是一种针对kNN分类方法的数据预处理算法。假设待分类的数据集D，对于其中每一个样本点x, 
使用另一kNN分类器在D-x上进行训练并对x进行预测。如果结果与x的实际结果一致，则保留x，否则剔除x。

在Under-sampling的语义下，可以直接对负样本点的点进行判断并去掉分类错误的点。从直观上来说，
这种方法相当于去除了noise/borderline中的点。  
另外一种利用ENN进行Under-sampling的作法是Repeated ENN, 相对于上一种方法，该方法重复进行ENN直到
没有分类错误的点或某类负样本点过少为止。

#### NCL (Neighborhood Cleaning Rule)
在原始的ENN方法中，是去除了负样本中分类错误中的点。在NCL中，更进一步的，对于分类错误的正样本，
去除掉其k个最近邻中的负样本点。这种方法直观上可以看做进一步去除掉导致分类错误的噪音。

![](/images/imbalanced_learning/8.png){: width="460px" height="170px"} 

对于第三步，在3-NN中，O中误分类C的样本放入$$ A_2 $$，同时为了避免过多的去除O中较小类的样本，所以
设置了O中误分类C的样本所属的类$$ C_i $$需要满足 $$ |C_i| \ge 0.5 \times |C| $$.

### (5) Cluster-Based Sampling Method
对于Undersampling，假如需要采k个点，这种方法将所有的负样本点聚类成K个簇，并将每个簇的中心做为采样的结果。
由于聚类的簇的个数是已知的，可以直接使用K-Means方法完成聚类。

对于Oversampling，可以使用K-Means先进行聚类。设多数类的类簇最大大小是max_cluster_size，先对于每个多数类类簇
随机采样直到大小等于max_cluster_size。设多数类总样本数为maxclass_size，对于每个少数类，随机采样直到每个类簇
包含maxclass_size/numofsmallclass的样本数。  
除了随机采样，还有一种方法是合成采样，对于k-means每次迭代中求出的所有中心点加入到样本集中。

### (6) Integration of Sampling and Boosting
一般是在Boosting的每次迭代中使用采样方法来过采样少数类样本。

#### SMOTEBoost
Adaboost同等对待FP和FN这两种错误，为了提升少数类样本的权值，提出了SMOTEBoost.  
SMOTEBoost结合了SMOTE和Adaboost.M2，在每一个boosting迭代中，引入了合成采样方法，增加少数类样本个数。

![](/images/imbalanced_learning/9.png){: width="520px" height="360px"} 

#### DataBoost-IM
设$$ M_L = min(|S_{maj}|/|S_{min}|, |E_{maj}|)，M_S = min(|S_{maj}|\times M_L / |S_{min}|，|E_{min}|) $$，
合成的样本集为$$ E_{syn} $$，并有$$ E_{smin} \subset E_{syn}，E_{smaj} \subset E_{syn} ， |E_{smin}|=M_S \times |S_{min}|，|E_{smaj}|=M_L \times |S_{maj}| $$，
其中$$ E $$为选出的难以分类的样本点的集合。

![](/images/imbalanced_learning/10.png){: width="400px" height="700px"}

## Cost-Sensitive Methods
采样考虑的是平衡不同类的样本的分布，是data-level的；代价敏感方法考虑的是误分类样本的代价，是algorithm-level的。
使用代价矩阵(cost matrix)表示误分类样本的代价。近年研究表明代价敏感学习与不均衡样本学习有很大的关联，因此代价敏感方法可以自然的适用到不均衡学习问题中。

三种方法：  
1. Weighting the data space  
2. Making  a  specific  classifier  learning  algorithm  cost-sensitive  
3. Using Bayes risk theory to assign each sample to its lowest risk class

## 参考资料
[Learning from Imbalanced Data](/docs/imbalanced_learning/1.pdf)  