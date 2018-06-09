---
layout: post
title:  "Parameter Server"
date:   2018-06-05 11:18:02
categories: DistributedComputing
tags: DistributedComputing
excerpt: Parameter Server
---

## 1. 简介

[ps-lite](https://github.com/dmlc/ps-lite)框架是DMLC实现的parameter server通信框架。整体架构如下

<img src="/images/ps/ps1.png" width="60%" height="60%">

有三种角色的node:

1. server: 保存和更新模型参数的权重，每个server node只存一部分模型。  
2. worker: 负责具体的计算逻辑。从server上拉取参数，并将参数更新推到server。  
3. scheduler: 调度器，控制其他node。

## 2. ps-lite代码实现

### 2.1 SArray

SArray类用`std::shared_ptr`实现的数组，用于替代`std::vector`，避免数组的深拷贝。

利用`std::shared_ptr`实现了零拷贝的构造函数、复制构造函数、赋值操作符。

其中`ptr_ = std::shared_ptr<V>(arr.ptr(), reinterpret_cast<V*>(arr.data()));`使用了如下的
构造函数。之所以不直接用构造函数或者赋值操作符，是因为要强转保存数据的类型为`V*`。

```cpp
//it allows you to create a shared pointer which returns one pointer through its get() function but maintains ownership of other, possibly unrelated pointer
template< class Y > 
shared_ptr( const shared_ptr<Y>& r, element_type* ptr ) noexcept;
```

```cpp
 /**
   * \brief construct from a c-array
   * Zero-copy constructor, namely just copy the pointer
   * \param data the source data
   * \param size the length
   * \param deletable whether or not can call `delete [] data` when the reference
   * count goes 0
   */
  SArray(V* data, size_t size, bool deletable = false) {
    if (deletable) {
      reset(data, size, [](V* data){ delete [] data; });
    } else {
      reset(data, size, [](V* data) { });
    }
  }
  /**
   * \brief construct from a shared std::vector pinter, no data copy
   */
  explicit SArray(const std::shared_ptr<std::vector<V>>& vec) {
    ptr_ = std::shared_ptr<V>(vec, vec->data());
    size_ = vec->size();
    capacity_ = size_;
  }
  /**
   * \brief construct from another SArray.
   * Zero-copy constructor, namely just copy the pointer
   * \tparam W the value type of the source array
   * \param arr the source array
   */
  template <typename W> void operator=(const SArray<W>& arr) {
    size_ = arr.size() * sizeof(W) / sizeof(V);
    CHECK_EQ(size_ * sizeof(V), arr.size() * sizeof(W)) << "cannot be divided";
    capacity_ = arr.capacity() * sizeof(W) / sizeof(V);
    ptr_ = std::shared_ptr<V>(arr.ptr(), reinterpret_cast<V*>(arr.data()));
  }
```

### 2.2 Resender

Resender类作为Van类中的一个对象，作用是发送请求后一段时间未收到ACK回复则重发请求。

Resender主要包含如下两个函数。  AddOutgoing作用是把将要发出的一些请求放到buffer里，Resender会定时发送buffer中的请求。  
AddIncomming则是接受ACK回复以及回复ACK。

```cpp
void AddOutgoing(const Message& msg);
bool AddIncomming(const Message& msg);
```


### 2.3 Van

Van是负责通信的类，建立节点直接之间的连接（Worker与Scheduler、PServer与Scheduler、Worker与PServer。开启本地的receiver_thread负责接收请求。

通过如下方式开启一个接收请求的线程：

```cpp
receiver_thread_ = std::unique_ptr<std::thread>(
            new std::thread(&Van::Receiving, this));
// Receiving函数定义如下
void Van::Receiving() {
  Meta nodes;
  Meta recovery_nodes;  // store recovery nodes
  recovery_nodes.control.cmd = Control::ADD_NODE;

  while (true) {
    Message msg;
    int recv_bytes = RecvMsg(&msg);
    // For debug, drop received message
    if (ready_.load() && drop_rate_ > 0) {
      unsigned seed = time(NULL) + my_node_.id;
      if (rand_r(&seed) % 100 < drop_rate_) {
        LOG(WARNING) << "Drop message " << msg.DebugString();
        continue;
      }
    }

    CHECK_NE(recv_bytes, -1);
    recv_bytes_ += recv_bytes;
    if (Postoffice::Get()->verbose() >= 2) {
      PS_VLOG(2) << msg.DebugString();
    }
    // duplicated message
    if (resender_ && resender_->AddIncomming(msg)) continue;

    if (!msg.meta.control.empty()) {
      // control msg
      auto& ctrl = msg.meta.control;
      if (ctrl.cmd == Control::TERMINATE) {
        ProcessTerminateCommand();
        break;
      } else if (ctrl.cmd == Control::ADD_NODE) {
        ProcessAddNodeCommand(&msg, &nodes, &recovery_nodes);
      } else if (ctrl.cmd == Control::BARRIER) {
        ProcessBarrierCommand(&msg);
      } else if (ctrl.cmd == Control::HEARTBEAT) {
        ProcessHearbeat(&msg);
      } else {
        LOG(WARNING) << "Drop unknown typed message " << msg.DebugString();
      }
    } else {
      ProcessDataMsg(&msg);
    }
  }
}
```


Van不仅负责通信，还保存了一些额外的信息，比如上面的`Receiving` 函数中的nodes、recovery_nodes等。以及成员变量`std::vector<int> barrier_count_`保存了同步相关的计数。

Van本身处理了如下几类Control Message(控制类的请求)，其余的Message（数据类的请求、BARRIER的回复等）则交给Customer  
```
enum Command { EMPTY, TERMINATE, ADD_NODE, BARRIER, ACK, HEARTBEAT };
```


### 2.4 Customer

Customer类记录每个request及其对应的response的个数。

```cpp
// 处理每个接收到的请求的函数（std::function<void(const Message& recved)>）
RecvHandle recv_handle_;
// 线程安全队列，存放接收的请求
ThreadsafeQueue<Message> recv_queue_;
std::unique_ptr<std::thread> recv_thread_;
// 成员变量tracker_记录每个请求可能发送给了多少节点以及从多少个节点返回。
std::mutex tracker_mu_;
std::condition_variable tracker_cond_;
std::vector<std::pair<int, int>> tracker_;
```

### 2.5 Postoffice

Postoffice是全局管理类，每个节点中都有一个该类的对象。记录了当前节点的状态信息，例如该节点的角色、worker和pserver个数等。计算worker/pserver rank到id的转换。

GetServerKeyRanges函数计算每个PServer对应的key的的范围。
PServer只支持int类型的key，所以该函数只需要平分int的范围。

```cpp
#if USE_KEY32
/*! \brief Use unsigned 32-bit int as the key type */
using Key = uint32_t;
#else
/*! \brief Use unsigned 64-bit int as the key type */
using Key = uint64_t;
#endif
/*! \brief The maximal allowed key value */
static const Key kMaxKey = std::numeric_limits<Key>::max();

const std::vector<Range>& Postoffice::GetServerKeyRanges() {
  server_key_ranges_mu_.lock();
  if (server_key_ranges_.empty()) {
    for (int i = 0; i < num_servers_; ++i) {
      server_key_ranges_.push_back(Range(
          kMaxKey / num_servers_ * i,
          kMaxKey / num_servers_ * (i+1)));
    }
  }
  server_key_ranges_mu_.unlock();
  return server_key_ranges_;
}
```

### 2.6 SimpleApp

一个简单的收发消息的类，里面含有一个Customer对象用来控制请求连接。

```cpp
// 构造函数里传入了SimpleApp::Process作为Customer的recv handle
inline SimpleApp::SimpleApp(int app_id, int customer_id) : SimpleApp() {
  using namespace std::placeholders;
  obj_ = new Customer(app_id, customer_id, std::bind(&SimpleApp::Process, this, _1));
}

// Customer接收到请求后，调用SimpleApp::Process处理请求
// 这里的request_handle_和response_handle_则是SimpleApp::Process中用于处理消息的函数
inline void SimpleApp::Process(const Message& msg) {
  SimpleData recv;
  recv.sender    = msg.meta.sender;
  recv.head      = msg.meta.head;
  recv.body      = msg.meta.body;
  recv.timestamp = msg.meta.timestamp;
  recv.customer_id = msg.meta.customer_id;
  if (msg.meta.request) {
    CHECK(request_handle_);
    request_handle_(recv, this);
  } else {
    CHECK(response_handle_);
    response_handle_(recv, this);
  }
}
```

### 2.7 KVWorker KVServer

一个key可能对应多个value，所以用如下数据结构存储

```cpp
template <typename Val>
struct KVPairs {
  // /** \brief empty constructor */
  // KVPairs() {}
  /** \brief the list of keys */
  SArray<Key> keys;
  /** \brief the according values */
  SArray<Val> vals;
  /** \brief the according value lengths (could be empty) */
  SArray<int> lens;
};
```

worker中的pull和push是异步的，会返回一个int类型的timestamp。
调用wait传入timestamp，可以阻塞等待同步。
或者可以在push或pull传入回掉函数Callback设置后续操作。

```cpp
int Push(const std::vector<Key>& keys,
           const std::vector<Val>& vals,
           const std::vector<int>& lens = {},
           int cmd = 0,
           const Callback& cb = nullptr) {
    return ZPush(
        SArray<Key>(keys), SArray<Val>(vals), SArray<int>(lens), cmd, cb);
  }
```


KVServer中通过如下的handle处理接收的请求，也可以自定义handle函数。
KVServerDefaultHandle函数维护一个哈希表，接收请求更新和查询key对应的value.

```cpp
template <typename Val>
struct KVServerDefaultHandle {
  void operator()(
      const KVMeta& req_meta, const KVPairs<Val>& req_data, KVServer<Val>* server) {
    size_t n = req_data.keys.size();
    KVPairs<Val> res;
    if (req_meta.push) {
      CHECK_EQ(n, req_data.vals.size());
    } else {
      res.keys = req_data.keys; res.vals.resize(n);
    }
    for (size_t i = 0; i < n; ++i) {
      Key key = req_data.keys[i];
      if (req_meta.push) {
        store[key] += req_data.vals[i];
      } else {
        res.vals[i] = store[key];
      }
    }
    server->Response(req_meta, res);
  }
  std::unordered_map<Key, Val> store;
};
```

## 3. 一些细节

### 3.1 worker和pserver处理消息的流程

worker和pserver启动时首先会执行Postoffice::Start函数，该函数初始化当前节点并调用了Van::Start连接scheduler，启动两个线程：接收消息的线程recv thread以及发送心跳的线程。

worker和pserver都继承自SimpleApp，所以类中都有一个Customer对象，
customer对象本身也会启动一个recv thread，其中调用注册的recv_handle_函数对消息进行处理。

对于worker来说，其注册的recv_handle_是KVWorker::Process()函数。因为worker的recv thread接受到的消息主要是从server处pull下来的KV对，因此该Process()主要是接收message中的KV对。

```cpp
template <typename Val>
void KVWorker<Val>::Process(const Message& msg) {
  if (msg.meta.simple_app) {
    SimpleApp::Process(msg); return;
  }
  // store the data for pulling
  int ts = msg.meta.timestamp;
  if (!msg.meta.push && msg.data.size()) {
    CHECK_GE(msg.data.size(), (size_t)2);
    KVPairs<Val> kvs;
    kvs.keys = msg.data[0];
    kvs.vals = msg.data[1];
    if (msg.data.size() > (size_t)2) {
      kvs.lens = msg.data[2];
    }
    mu_.lock();
    recv_kvs_[ts].push_back(kvs);
    mu_.unlock();
  }

  // finished, run callbacks
  if (obj_->NumResponse(ts) == Postoffice::Get()->num_servers() - 1)  {
    RunCallback(ts);
  }
}
```

对于pserver来说，其注册的recv_handle_是KVServer::Process()函数。因此pserver接受的是worker们push上来的KV对，需要对其进行处理，因此该Process()函数中调用的用户通过KVServer::set_request_handle()

```cpp
template <typename Val>
void KVServer<Val>::Process(const Message& msg) {
  if (msg.meta.simple_app) {
    SimpleApp::Process(msg); return;
  }
  KVMeta meta;
  meta.cmd       = msg.meta.head;
  meta.push      = msg.meta.push;
  meta.sender    = msg.meta.sender;
  meta.timestamp = msg.meta.timestamp;
  meta.customer_id = msg.meta.customer_id;
  KVPairs<Val> data;
  int n = msg.data.size();
  if (n) {
    CHECK_GE(n, 2);
    data.keys = msg.data[0];
    data.vals = msg.data[1];
    if (n > 2) {
      CHECK_EQ(n, 3);
      data.lens = msg.data[2];
      CHECK_EQ(data.lens.size(), data.keys.size());
    }
  }
  CHECK(request_handle_);
  // 用户可以自定义pserver中对kv处理的函数
  request_handle_(meta, data, this);
}
```

### 3.2 同步阻塞

worker可以调用如下函数等待pull或者push操作的完成，其中obj_即为worker中的Customer对象。

worker中的`std::vector<std::pair<int, int>> tracker_`变量记录了request个数及其对应的response个数。

tracker_cond_是条件变量，Wait函数会一直阻塞直到request个数与response个数相等。


```cpp
void KVWorker::Wait(int timestamp) { 
    obj_->WaitRequest(timestamp); 
}

void Customer::WaitRequest(int timestamp) {
  std::unique_lock<std::mutex> lk(tracker_mu_);
  tracker_cond_.wait(lk, [this, timestamp]{
      return tracker_[timestamp].first == tracker_[timestamp].second;
    });
}
```

### 3.3 位运算

worker、server、scheduler属于三个group,分别用数字4、2、1表示id，即100、010、001。

这样如果要表示worker group和server group,　id即为100+010=110。
这样只需将目标节点id设为4即可向所有worker发送请求，设为6则是向所有worker和pserver发送请求。因此1-7内任意一个数字都代表的是Scheduler/Server/Worker的某一种组合。

单个节点的id从8开始，8,10,12...表示worker0,worker1,worker2...（即2*n+8）  

9,11,13...表示server0,server1,server2（即2*n+9）


## 4. 参考资料
[Parameter Server for Distributed Machine Learning](/docs/parameter_server/ps.pdf)  
[Scaling Distributed Machine Learning with the Parameter Server](/docs/parameter_server/Scaling Distributed Machine Learning with the Parameter Server.pdf)