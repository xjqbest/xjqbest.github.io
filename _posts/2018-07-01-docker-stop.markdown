---    
layout: post
title:  "docker stop时上传日志"
date:   2018-07-01 11:00:00
categories: Docker
tags: Docker
excerpt: 
---

## 1 背景

假如我们现在使用docker跑一些作业，
现在有一个需求是在我们停掉docker时（即执行docker stop）,
需要将容器内的一些日志先上传到hdfs。

## 2 解决办法

### 2.1 概述

docker有优雅退出机制，就是容器停止的时候，会给容器内主进程发一个TERM信号。
因此可以在收到该信号后，上传日志。当然执行docker stop后是有个超时时间的，
超过时间未上传完也会被kill。

### 2.2 docker stop

docker stop和docker kill分别实现了优雅退出和强行退出两个操作：  
1. docker stop：向容器内1号进程，发送SIGTERM信号，在一定时间之后（可通过参数指定）再发送SIGKILL信号。
2. docker kill：直接发送SIGKILL信号。

### 2.3 需要注意的地方

为了确保进程能接收到SIGTERM信号，需要注意如下两点

#### 2.3.1 CMD格式

Dockerfile中CMD需要采用如下的exec格式，否则接收不到信号  
CMD ["executable","param1","param2"]

其中CMD有三种格式： 
1. CMD ["executable","param1","param2"]（exec 格式, 推荐使用这种格式）
2. CMD ["param1","param2"] （作为 ENTRYPOINT 指令参数）
3. CMD command param1 param2 （shell 格式，默认 /bin/sh -c），使用 shell 格式之后，程序会以 /bin/sh -c 的子命令启动，并且 shell 格式下不会传递任何信号给程序。

#### 2.3.2 容器入口脚本

在容器中使用自定义bash脚本作为容器入口，脚本中使用后台方式执行具体应用的命令，然后使用内建wait阻塞。需要注意进程处理信号的限制：只有当进程阻塞在内建命令时才可以响应SIG信号，否则会一直等待子进程退出后再处理。

内部命令实际上是shell程序的一部分，shell不需要创建子进程，比如：exit，history，cd，echo，wait，trap等，linux系统加载运行时shell就被加载并驻留在系统内存中，一般不会fork子进程。
外部命令是linux系统中的实用程序部分，需要时才将其调用内存。一般会fork出子进程。
用type命令可以分辨内部命令与外部命令。


如果脚本写成如下这样，那么就会导致docker stop后，等到10000秒之后才能处理SIGTERM：
```
#！/bin/bash
trap 'exit 0' SIGTERM
sleep 10000s
```

## 3 小实验

### 3.1 脚本

Dockerfile
```
FROM registry.hub.docker.com/library/centos:latest
WORKDIR /root/mytest
ADD entry.sh /root/mytest/entry.sh
ADD user_script.sh /root/mytest/user_script.sh
RUN chmod +x /root/mytest/user_script.sh && \
    chmod +x /root/mytest/entry.sh
CMD ["/bin/bash", "-c", "/root/mytest/entry.sh"]
```


user_script.sh
```bash
while true
do
    echo "doing user_script"
    sleep 3s
done
```


entry.sh
```bash
#！/bin/bash
trap 'echo "before exiting" && sleep 5s && echo "exiting" && exit 0' SIGTERM
bash -c /root/mytest/user_script.sh &
wait $!
```

这里用sleep 5s代表上传日志。

### 3.2 执行的命令

（1）先build镜像： docker build -t test_stop_graceful:latest .

（2）run该镜像：   docker run -it --rm test_stop_graceful:latest

（3）然后在某一时刻执行 docker stop:    docker stop xxxxx


### 3.3 实验结果

<img src="/images/docker_stop/1.png" width="25%" height="25%">