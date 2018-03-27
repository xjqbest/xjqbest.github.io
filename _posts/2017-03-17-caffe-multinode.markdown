---
layout: post
title:  "caffe multinode节点间通信代码阅读"
date:   2017-03-17 23:15:00
categories: C++ DistributedComputing
tags: C++ DistributedComputing
excerpt: 
---

## caffe简介

> Caffe is a deep learning framework made with expression, speed, and modularity in mind.

官网:[http://caffe.berkeleyvision.org/](http://caffe.berkeleyvision.org/) 

github地址：[https://github.com/BVLC/caffe](https://github.com/BVLC/caffe)

## multinode版caffe

多机版的caffe，使用[zeromq](http://zeromq.org/)作为异步通信，同时采用数据并行和模型并行。

github地址: [https://github.com/01org/caffe/tree/multi_node](https://github.com/01org/caffe/tree/multi_node)

## 总体结构

深度学习神经网络通常是前面几个卷积层后面再加上全连接层。由于全连接层参数较大，所以可以在FC层做模型并行。  
因此整体上分为两大部分，前半部分做卷积，后半部分做全连接。

![](/images/caffe/1.png){: width="400px" height="300px"}

每个节点扮演一个或多个角色，有：

（1）Convolution Node

这部分包括卷积层等，多个Convolution，每个分到一部分数据，实现了数据并行。

（2）FC Node

这部分包括全连接层等，多个FC，每个分到一部分参数，实现了模型并行。

（3）Parameter Server Node
用来更新每个Convolution得到的梯度。可以有多个PS，每个只有一部分参数。

（4）Model Server Node
用来解析网络所形成的DAG图，根据配置文件把一部分图结构分给PS，另一部分分给FC。

## 源码分析

### Msg类

把消息和相关信息封装成类，是节点间消息传递的基本单位。  
包含源地址、目的地址、消息id、消息类型、存储的每一个blob的信息、convolution thread的id、client的clock、是否只包含部分信息等等。  
包含以下成员变量：

```cpp
MsgHeader header_; // MsgHeader包含src、dst、msg_id、type、BlobInfo、conv_id、clock、data_offset、is_partial
vector<zmq_msg_t *> zmsg_vec_; //所有消息数据的指针
map<string, int> blob_name_index_; // blob name to index
```

部分成员函数：

```cpp
void add_blob_info(const BlobInfo& b); // 往header里添加一个BlobInfo，同时更新blob_name_index_
int AppendZmsg(zmq_msg_t *zmsg); // 往zmsg_vec_添加一个消息数据指针
int MergeMsg(shared_ptr<Msg> m); // 合并新来的m, 把m包含的所有blob和其对应的消息数据加进来。
shared_ptr<Msg> Msg::ExtractMsg(const string& blob_name); // 把blob_name对应的blob移除。
void Msg::AddNewBlob(const string& blob_name, const void *data, int sz); // 添加一个新的blob，数据起始地址为data，size是sz
void Msg::CopyBlob(const string& blob_name, void *data, int sz); // 把blob_name对应的所有消息数据copy到data所指的地址。
```

### SkSock类和SkServer类

封装了Msg的收发，其中SKServer封装了ZMQ_ROUTER类型socket的收发。  
SkSock部分成员变量：

```cpp
void *sock_; // the native zmq socket
string addr_; // the zmq address of this socket
int id_; // an unique id allocated by an global id server
int sk_type_; // an unique id allocated by an global id server
static boost::mutex zmq_init_mutex_; // a zmq context should be inited only once
static int inited_cnt_;
```

### MsgHub类

异步收发消息的类，是Conv node、FC node、PS node、test node的基类。

首先看一看成员变量：

```cpp
int nthreads_; // 总线程数
int nworkers_; // 用来作为worker的线程数
vector<shared_ptr<WorkerThread<Dtype> > > threads_;
vector<shared_ptr<SkSock> > sockp_arr_; // ZMQ_PAIR类型的socket，与进程内线程通信
vector<shared_ptr<SkSock> > prior_socks_; // 只负责与worker通信的ZMQ_PAIR socket （send packets to worker with priority）
zmq_pollitem_t *poll_items_; // message polling
int num_poll_items_;
string node_ip_;
CPUDispatcher dispatcher_;
shared_ptr<SkSock> sock_pub_; // ZMQ_PUB, 向下游节点广播blob (broadcast blobs to downstream nodes)
shared_ptr<SkSock> sock_back_; // ZMQ_ROUTER, a ROUTER socket to received the packets from downstream nodes
int back_sock_index_; // back sock index in the poll table,  back_sock_index_ = nthreads
unordered_map<int, shared_ptr<SkSock> > node_to_sock_; // map from node id to the sock index
int node_id_;// NodeID, fetched from ID server
int param_thread_index_; // param_thread_index_ = nthreads - 1;
vector<shared_ptr<SkSock> > vec_sub_sock_; // ZMQ_SUB,  receive broadcast message from upstream nodes
int sub_sock_index_; // sub_sock_index_ = back_sock_index_ + 1
int poll_offset_; // num_poll_items_ = poll_offset_ + 1;
```

部分成员函数：

```cpp
virtual int Init() = 0; // 初始化
virtual int RouteMsg() = 0; // 节点对收到每个消息转发到相应节点/线程中去。
virtual int SetUpPoll(); // 设置poll_items_
int Poll(); // 先调用SetUpPoll()初始化poll_items_，然后循环调用zmq_poll和RouteMsg来接收消息
int StartThreads(); // 开始所有线程
void Enqueue(int thrd_id, shared_ptr<Msg> m); // 把消息发给一个id为thrd_id的worker thread
void MsgHub<Dtype>::InitRoute(); // 连接上游类型为PUB的socket用来接受消息，连接上游类型为ROUTER的socket用来发送消息
shared_ptr<SkSock> ConnectNode(const string& addr, int dst_id); // 使用一个ZMQ_DEALER类型的socket连接addr
```

### WorkerThread

是convolution thread, fc thread, ps thread的基类。

待补充

### PSThread

ps thread接收来自conv node的梯度，并使用随机梯度下降的方法更新参数，并将更新后的参数返回给conv node。

成员变量：

```cpp
  // 使用随机梯度下降的方法更新参数
  SGDSolver<Dtype> *ps_solver_;
  // 当前迭代次数
  int iter_;
  // map from client id to its array index
  map<int, int> client_idx_map_;
  // zmq id of each client
  vector<int> client_ids_;
  // store the messages from clients
  vector<vector<shared_ptr<Msg> > > client_msgs_;
  // number of gradients updated
  int updated_layers_;
  // number of learnalbe layers
  int num_learnable_layers_;
  // clock of each message
  vector<vector<int> > msg_clocks_;
  // allowed staleness of PS， -1 means doesn't check staleness
  int staleness_;
  // the number of conv. workers，即conv node的个数
  int num_workers_;
  // maximun iterations to be executed，最大迭代次数
  int max_iter_;
  // the id of test node
  int test_node_;
  // 发来注册消息的conv node个数
  int registered_workers_;
```

成员函数：

```cpp
int SendUpdates(int layer_id); // 找到该layer对应的clock最小的msg，并更新梯度后，将该layer的参数返回给conv node
void UpdateLayer(int layer_id); // 更新一个layer的参数
void BroadcastLayer(int layer_id); // 将该layer的参数返回给所有conv node
void RegisterNode(shared_ptr<Msg> m); // 注册每一个conv node，并将ps的参数返回给每一个conv node
int UpdateParam(shared_ptr<Msg> m); // 每次等所有的conv node的相同clock的msg都来了之后，更新该层的参数
void SendParam(shared_ptr<Net<Dtype> > net, const vector<string>& layer_names,int dst, int clock); // 将制定的layers的参数发回给conv node
```

### ModelMap

每个Conv node, PS node, FC node, test node初始化的时候会向Model Server发送GET_TRAIN_MODEL类型的消息从而获得自己的那部分layer.

ModelMap 负责解析发来的请求，检查图的结构，并回给相应节点模型信息和route信息，并把ps node 和 fc node的地址告诉conv node.

ModelServer开了两个线程：  
（1）id server线程用来给每个node分配id  
（2）model server线程用来接收类型为GET_TRAIN_MODEL的消息，并调用ModelMap  

从ModelServer的功能看出，待其他所有node初始化完成后，它就没有用了。

先看一下成员变量

```cpp
 enum Status {
    WAIT_REQUESTS = 0,
    WAIT_FC_GATEWAY,
    // wait for conv nodes to complete
    INITED,
    COMPLETED
  }; // 发来请求后，根据当前状态来更新新的状态
  Status status_;
  // we use the full solver to generate layers
  Solver<Dtype> *psolver_;
  SolverParameter solver_param_; // solver prototxt的配置参数
  // clear solver parameter without layers, only have solver param
  SolverParameter clear_solver_;
  // solver for conv. nodes
  SolverParameter conv_solver_;
  // solver parameter for test nodes
  SolverParameter test_solver_;
  shared_ptr<Net<Dtype> > net_;
  NetParameter net_param_; // train_test prototxt中的网络配置（过滤掉了TEST）
  // all the layers
  vector<LayerParameter> layer_params_;
  //
  vector<bool> layers_filled_; // bfs中是否遍历layer index对应的layer
  // number of inputs of the layer
  vector<int> layer_inputs_;
  // map of layer names
  map<string, int> layer_name_idx_;
  // net forward graph
  vector<vector<int> > net_forward_graph_; // layer之间的关系
  // layers of the net is modeled as a graph
  vector<vector<int> > net_backward_graph_; // layer之间的关系
  // sub net forward graph
  vector<vector<int> > sub_forward_graph_;
  // sub net backward graph
  vector<vector<int> > sub_backward_graph_;
  // layers in a sub graph (sorted in BFS)
  vector<vector<int> > sub_solver_layers_;  // 第一维是layer的index。每个request包含的所有layer。
  // the name of sub layers
  vector<vector<string> > sub_solver_layer_names_; // 相应的name
  // input, output and forward blobs are used for routing
  // name of input blobs for a sub graph
  vector<vector<string> > sub_input_blobs_; // 第一维是layer的index（唯一标识了node/request）。表示所拥有的输入blob
  vector<vector<string> > sub_output_blobs_; // 表示所拥有的输出blob
  vector<int> conv_fwd_nodes_;
  // indices to parameter server nodes
  vector<int> ps_nodes_; // 存的是相应request_[] 的索引
  // indices to FC nodes
  vector<int> fc_nodes_; // 存的是相应request_[] 的索引
  vector<int> fc_gateways_;
  // output nodes
  vector<int> output_nodes_;
  // store all the route nodes
  vector<vector<RouteNode> > route_nodes_; // 第一维是layer的index， 第二维是split。 （每个split包含所有的layer）
  // store the model requests in a 2D vector
  vector<vector<shared_ptr<ModelRequest> > > requests_; // 第一维是layer的index， 第二维是split的position
  // whether the requests of this layer is full filled
  vector<bool> request_filled_;  // request是否填满 （第一维是layer的index）
  // whether the request is parsed
  vector<bool> request_parsed_; // request是否已被解析 （第一维是layer的index）
  // store the conv client request from data parallel cliens
  vector<shared_ptr<ModelRequest> > conv_requests_; // 从conv client 收到的request
  // requests from testing nodes
  vector<shared_ptr<ModelRequest> > test_requests_;
  // the generated message for FC layers
  vector<shared_ptr<Msg> > replies_;
  int fc_batch_size_;
  int num_workers_; // conv client 数
  // number of overlapping sub solvers
  int num_sub_solvers_;
  // routing info for parameter servers
  vector<RouteInfo> ps_routes_; // 所有ps的splits
  // routing info for conv nodes
  vector<RouteInfo> conv_routes_;
```

成员函数：

```cpp
// 构造函数，从配置文件中读取solver和net的参数，并且把data layer的batch设为batch_size / num_sub_solvers_（即数据并行）
ModelMap(const string full_solver/*solver prototxt*/, int nworkers/*fc worker num*/, int sub_solvers/*number of sub-solver in conv*/);
// 初始化net相关变量,根据网络结构构造前向传播的图net_forward_graph_和后向传播的图net_backward_graph_
void ModelMap<Dtype>::BuildNetGraph();
// 得到每个ps node 和 fc node的输入blob(sub_input_blobs_)和输出blob(sub_output_blobs_)
void ModelMap<Dtype>::ParseInputOutput(int node_idx);
// 把编号为node_idx的node的所有layer添加进pnet所指的网络，并且把全连接层输出设为num_output / num_splits（即模型并行）
void ModelMap<Dtype>::AddLayers(NetParameter *pnet, int node_idx);
// 把编号为node_idx的node的所有input blob加入到pnet所指的网络
void ModelMap<Dtype>::AddInputs(NetParameter *pnet, int node_idx);
// 初始化编号为node_idx的node的solver（分别调用AddLayers和AddInputs），设置RouteInfo的solver，worker数目，worker内流水线数目。
void ModelMap<Dtype>::AddSolver(RouteInfo *proute, int node_idx);
// 用sub_forward_graph_[node_idx]的所有节点初始化bcast_nodes，用sub_backward_graph_[node_idx]的所有节点初始化prev_nodes
void ModelMap<Dtype>::AddRoutes(RouteInfo *proute, int node_idx);
// 设置返回给FC node的消息（调用AddSolver和AddRoutes， 并设置fc_gateway的地址）
int ModelMap<Dtype>::PrepareFCMsg();
// 设置返回给PS node的消息 
int ModelMap<Dtype>::PreparePSMsg(); 
// 设置返回给test node的消息。设置了ps node和fc node的地址
int ModelMap<Dtype>::PrepareTestMsg();
// 设置返回给Conv node的消息
int ModelMap<Dtype>::PrepareConvMsg();
// 把ps node的solver中的所有layer添加给conv_solver_。这说明了conv是给ps干活儿的，ps负责接收conv算好的结果和反向更新参数。
void ModelMap<Dtype>::PrepareConvSolver();
// 首先调用ParseInputOutput得到ps node和fc node的输入blob、输出blob。然后利用利用fc node的输入blob建立node之间的sub_forward_graph_和sub_backward_graph_。并且可以从fc node中区别开gateway node（需要和conv node 通信的fc node）和forward node
int ModelMap<Dtype>::PrepareRoutes()
// 使用bfs从start layer到end layers结束，把layer加入到route_nodes_[start layer index][every split pos]
void ModelMap<Dtype>::ParseRequest(int start_idx);
// 等待所有的split后的position齐了以后, 调用ParseRequest解析请求
void ModelMap<Dtype>::AddModelRequest(shared_ptr<ModelRequest> rq);
// 设置ps node、fc gateway node、fc forward node的地址。调用PrepareConvSolver设置conv_solver_.
void ModelMap<Dtype>::SetUpConvRoutes();
// 设置ps node的solver和nodeinfo
void ModelMap<Dtype>::SetUpPSRoutes();
// 当只有一个ps node，可以设置成树形的消息传递方式
int ModelMap<Dtype>::BuildReduceTree();
// 等所有conv node的请求都来了之后，初始化ps node的树形结构，并初始化会送给各个node的Msg
int ModelMap<Dtype>::UpdateWorkers();
// 等所有conv node的请求都来了之后，调用PrepareTestMsg初始化回送给test node的Msg。UpdateWorkers中的中也有调用PrepareTestMsg()，可以看出，启动ps、fc、test不需要按固定的顺序。
int ModelMap<Dtype>::ProcessTests(shared_ptr<Msg> m);
// 等所有conv node的请求都来了之后，调用UpdateWorkers
int ModelMap<Dtype>::ProcessConv(shared_ptr<Msg> m);
// 等所有的node的请求都来齐了之后(包括ps的所有split等等)，首先调用ParseInputOutput得到每个node的输入输出blob，并据此初始化sub_forward_graph_和sub_backward_graph_，并将fc node分为gateway node和forward node，然后调用PrepareFCMsg和UpdateWorkers来设置回送消息。
int ModelMap<Dtype>::ProcessModels(shared_ptr<Msg> m);
// 每当接收到一个node的消息时候，根据节点的类型调用ProcessConv或ProcessModels或ProcessTests.
int ModelMap<Dtype>::GetModel(shared_ptr<Msg> m);
```

### ParamServer

ParamServer就像它的名字一样，负责保存并更新参数。

conv node把计算的结果传给ps node，ps node更新梯度。

其中对于ps threads，实际只会用到一个。
