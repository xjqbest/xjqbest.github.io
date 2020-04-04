---
layout: post
title:  "常见深度学习框架调研-horovod"
date:   2020-04-01 10:05:00
categories: DistributedComputing
tags: DistributedComputing
excerpt: 常见深度学习框架调研-horovod
---

# horovod

[https://github.com/horovod/horovod](https://github.com/horovod/horovod)

#### 优点：
 - 与fleet一样 ，100%的原生api组网 + 分布式框架的Optimizer ，以及一些分布式框架相关的接口，例如hvd.hook
 - 最主要的特点是外层python封装很好，例如分布式任务提交、节点间通信等。
 - 支持mpi和gloo，支持timeline。
 - 创新点：批处理allreduce、autotune

#### 缺点：
 - 无大规模稀疏解决方案
 - 本身无pserver模式，计算逻辑全在backend里（好处是轻量级）。
 - 只是在DistributedOptimizer中，在更新梯度前添加all_reduce平均梯度。也可以看出，这个框架主要支持GPU训练。
 - 对流式训练支持的不好。


#### 与paddle fleet区别是什么
horovod主要支持GPU，fleet更灵活一点。并且对流式训练支持的更好。

#### all-reduce
为ring all reduce（Scatter-Reduce + Allgather），优点是通信总量只跟参数总量相关，与GPU卡数无关。并且吞吐量随GPU卡数线性增长。


#### 为何参数服务器基本都是cpu的，而gpu是同步的

参数服务器既可以用于cpu，也可以用gpu。

对于gpu而言，卡和卡之间通信是用GPUDirect+RDMA，不经过gpu-cpu内存-socket，速度也比较块。适合卡和卡之间做同步模式的训练。

pserver适用于卡和卡之间等待时间比较长，比如由于输入不均匀等原因，卡和卡之间通常同步较慢，异步训练的话就比较快。

如果将pserver模式引入gpu训练，可以配合流水线并行，避免gpu卡出现较长的空闲时间：

（1）卡和卡之间数据并行，卡内部做异步流水线，卡之间周期性allreduce梯度，并发sparse梯度给参数服务器。其中对于dense参数可以放在GPU内进行学习并做周期性同步。

（2）卡和卡之间模型并行，那就是卡和卡之间的异步流水线。

server端也可以搞流水并行：参数从ssd→ 内存→ GPU

#### 什么是scaling efficiency

<img src="/images/frameworks/hvd1.png" width="110%" height="110%">

#### tensor fusion

[https://github.com/horovod/horovod/blob/master/docs/tensor-fusion.rst](https://github.com/horovod/horovod/blob/master/docs/tensor-fusion.rst)

#### 批处理allreduce

是把多个tensor的allreduce合成一个来做，先把所有tensor拷到一个buffer里再做allreduce。减少通信次数，进而提升性能。

#### autotune

前几轮找出最优的超参组合，这里只考虑影响性能的那些超参，而不会影响模型的收敛效果。

[https://github.com/horovod/horovod/blob/master/docs/autotune.rst](https://github.com/horovod/horovod/blob/master/docs/autotune.rst)

#### 基本用法

可以看出支持原生api组网，不需要像xdl一样，定义了类似xdl.embedding之类的layer。
与fleet一样：100%的原生api组网 + 分布式框架的Optimizer ，以及一些分布式框架相关的接口，例如这里的hvd.hook

```python
import tensorflow as tf
import horovod.tensorflow as hvd
 
 
# Initialize Horovod
hvd.init()
 
# Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())
 
# Build model...
loss = ...
# Effective batch size in synchronous distributed training is scaled by the number of workers.
# An increase in learning rate compensates for the increased batch size.
opt = tf.train.AdagradOptimizer(0.01 * hvd.size())
 
# Add Horovod Distributed Optimizer
opt = hvd.DistributedOptimizer(opt)
 
# Add hook to broadcast variables from rank 0 to all other processes during
# initialization.
hooks = [hvd.BroadcastGlobalVariablesHook(0)]
 
# Make training operation
train_op = opt.minimize(loss)
 
# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None
 
# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)
```
