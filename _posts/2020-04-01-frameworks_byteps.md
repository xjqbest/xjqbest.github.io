---
layout: post
title:  "常见深度学习框架调研-byteps"
date:   2020-04-01 10:04:00
categories: DistributedComputing
tags: DistributedComputing
excerpt: 常见深度学习框架调研-byteps
---

# BytePS

[https://github.com/bytedance/byteps](https://github.com/bytedance/byteps)

#### byteps的应用场景

你有一个大规模的GPU集群，同时又有着包含大量的CPU、带宽资源的大池子。



#### 为什么比allreduce模式要好

比如有1MB数据，n个节点，使用ring all reduce，每个worker需要发送和接收2\*（n-1）／n，

平均每个worker约为2。而使用ps模式，每个worker只需发送1MB和接收1MB，通信总量变为allreduce的一半。

（但是也多用了n个cpu机器作为pserver）

在GPU集群中（无只有cpu的机器），如果使用pserver模式，就是一半worker一半server（因为带宽对等会更快），server端的GPU资源没有利用起来。


#### 优势


根据亚马逊云的价格，在GPU集群，额外购买同等数量的CPU机器作为server，只需要多花2%的钱，就有100%的瓶颈带宽提升。


#### 架构

另外byteps想说明，ps模式跟allreduce一样通用。并且ps相比allreduce，worker之间可以更好的异步训练。


byteps的架构图，plugin是各个深度学习计算框架，并将通信作为task放到优先级队列中，由byteps core处理通信。


<img src="/images/frameworks/byteps1.png" width="80%" height="80%">


#### 通信有哪些优化：

（1）前面的layer通信优先级更高，因为前向是从前往后，让下一轮的前向和上一轮反向梯度的通信做overlap

（2）切分大参数，合并小参数。不会因某一层参数过大而阻塞，不会因参数过小而由于发送过于频繁增大通信开销。


byteps使用了ps-lite 作为参数存取和通信（pull/push）的server，并且server端只保存梯度，不做参数更新，

把参数更新放在了worker端，好处是适用于所有框架，因为对于不同框架，对于同一种优化算法，实现可能都是不一样的。以及一些用户自定义的优化方法等。

byteps支持tensor partition 和 pipeline，将tensor切分成若干份，第一份执行完push梯度后，会执行pull操作，
与此同时执行第二份的push操作。其他份也是如此。充分利用双向带宽。


#### 计算流程

<img src="/images/frameworks/byteps2.png" width="100%" height="100%">


#### cross barrier

将iteration之间的global barrier去掉，变为layer-wise的依赖同时保证计算顺序的正确性。

<img src="/images/frameworks/byteps3.png" width="70%" height="70%">
