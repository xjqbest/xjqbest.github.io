---
layout: post
title:  "常见深度学习框架调研-xdl"
date:   2020-04-01 10:10:00
categories: DistributedComputing
tags: DistributedComputing
excerpt: 常见深度学习框架调研-xdl
---

# xdl

[https://github.com/alibaba/x-deeplearning](https://github.com/alibaba/x-deeplearning)

模型计算分为稀疏和稠密两部分，稀疏部分通过参数服务器，GPU加速，参数合并等技术极大提升了稀疏特征的计算和通信性能。稠密部分采用多backend设计，支持TF和Mxnet两个引擎作为计算后端，并且可以使用原生TF和Mxnet API定义模型。

xdl有大规模稀疏解决方案，实现了分布式embedding（存取、更新） 以及优化器，通过装饰器把用户调用tf和xdl的api定义的模型，添加反向op，以及xdl的通信op（push sparse，pull sparse，pull dense， push dense）。


#### 优点

**组网**

 - 支持多个backend，支持原生api组网，支持大规模稀疏。
 - 并且易用性很好，单机转分布式：使用xdl.embedding + xdl.optimize + xdl.session等。
 - 计算图的计算过程托管给了backend，因此支持backend本身支持的所有硬件。

**数据**

 - 读数据作为op，减少了部分接口（load/release等），c++端读数据处理数据。通过xdl.py_func定义op，可以从内存中读数据。

**创新点**

 - 提出Advanced Model Server，将worker中一部分前向反向计算放到server中，针对图像特征可以节省带宽。
 - 支持流式训练：[save delta](https://github.com/alibaba/x-deeplearning/wiki/%E6%A8%A1%E5%9E%8B%E5%A2%9E%E9%87%8F%E5%AF%BC%E5%87%BA)、[delete after unseen day](https://github.com/alibaba/x-deeplearning/wiki/%E7%89%B9%E5%BE%81%E6%B7%98%E6%B1%B0)、[create ratio](https://github.com/alibaba/x-deeplearning/wiki/%E7%89%B9%E5%BE%81%E5%87%86%E5%85%A5)、[dump](https://github.com/alibaba/x-deeplearning/wiki/XDL-Trace)
 - ps动态扩容支持：通过schedular模块实现。 

**缺点**

 - 数据：不支持pipe command。data parser定制化（读取pb格式、读取txt格式.....）。二次构建样本得添加op（add_op）。不支持全局shuffle。get batch op的输出定制化。

 - 性能：没有针对sparse和dense的通信做区分，没做多个batch之间本地merge梯度，pull sparse没做feasign去重。


**优化**

 - 请求合并：计算图中无依赖的通信节点，进行合并，减少通信次数，提高通信效率

 - 参数统一存储和平均分配

 - 针对低并发/单个batch含有超多特征的场景做了优化，XDL1.0侧重吞吐优化，采用one request per thread处理模型，能显著提高超高并发下的极限吞吐。但是在某些低并发/单个batch含有超多特征的场景下，这种单线程的处理方式会显著增加延迟。XDL1.2采用one request per thread及独立工作线程池两种处理模式，通过识别请求的特征数量自动选择合适的处理模式，达到吞吐和延迟兼顾的效果。


**这里说一下Advanced Model Server（AMS）在CrossMedia中的应用**

CrossMedia旨在应用图像信息对广告和用户行为进行理解和表达，帮助电商广告点击率（CTR）预估。主要创新体现在其训练系统。由于图像容量高，在以十亿计大规模数据引入图像特征（尤其是用户行为图像）将带来计算、存储、通信的沉重负担。

因此，XDL专门开发了Advanced Model Server（AMS）处理图像Embed网络。

AMS在以参数服务器（PS）为代表的分布式计算架构基础上再前进一步，将模型训练的一部分网络由worker部分移到了Server部分，也就是说，改造后的Server 也具有模型前向、后向计算的功能。

**在CrossMedia中**：

 - 图像和其他ID参数一样，分布式存储在Server里边，Worker需要时向Server请求，避免重复存储图像

 - Embed网络计算也放在Server里边（AMS的特有功能），图像被请求后，就地计算Embed，得到embed 向量传到Worker，这样的改造减少了Worker与Server的通讯，也去掉了同一Batch中请求同一图像的重复Embed计算

值得注意的是，AMS也可以处理各种类似的需要Embed的内容特征，如文本、视频。


### 代码阅读

数据IO，通过pybind封装c++端的DataIO类。

```python
import xdl
 
reader = xdl.DataReader("r1", # reader名称                 
                        paths=["./data.txt"], #文件列表
                        enable_state=False) # 是否打开reader state，用于分布式failover，开启的时候需要额外的命令行参数(task_num)
reader.epochs(1).threads(1).batch_size(10).label_count(1)
reader.feature(name='sparse0', type=xdl.features.sparse)\  # 定义reader需要读取的特征，本例包括两个sparse特征组和一个dense特征组
      .feature(name='sparse1', type=xdl.features.sparse)\
      .feature(name='deep0', type=xdl.features.dense, nvec=256)
reader.startup()
 
def train():
    batch = reader.read()
    sess = xdl.TrainSession()
    emb1 = xdl.embedding('emb1', batch['sparse0'], xdl.TruncatedNormal(stddev=0.001), 8, 1024, vtype='hash')
    emb2 = xdl.embedding('emb2', batch['sparse1'], xdl.TruncatedNormal(stddev=0.001), 8, 1024, vtype='hash')
    loss = model(batch['deep0'], [emb1, emb2], batch['label'])
    train_op = xdl.SGD(0.5).optimize()
    log_hook = xdl.LoggerHook(loss, "loss:{0}", 10)
    sess = xdl.TrainSession(hooks=[log_hook])
    while not sess.should_stop():
        sess.run(train_op)
```

对原始数据格式有要求，不支持pipe command，不支持分布式全局shuffle，支持failover。get batch op的输出很定制化，不易扩展：

```python
XDL_DEFINE_OP(GetBatch)
  .Attr("ds", AttrValue::kString)
  .Attr("sparse_count", AttrValue::kInt)
  .Attr("dense_count", AttrValue::kInt)
  .Attr("indicator_count", AttrValue::kInt)
  .Attr("dtype", AttrValue::kDataType)
  .Attr("tag_cnt", AttrValue::kInt, 0)
  .OutputList("indicators", DataType::kInt32, "indicator_count")
  .OutputList("indices", DataType::kInt32, "sparse_count")
  .OutputList("ids", DataType::kInt64, "sparse_count")
  .OutputList("segments", DataType::kInt32, "sparse_count")
  .OutputList("svalues", "dtype", "sparse_count")
  .OutputList("dvalues", "dtype", "dense_count")
  .OutputList("sample_indices", DataType::kInt32, "sparse_count")
  .OutputList("sample_segments", DataType::kInt32, "sparse_count")
  .Output("skbuf", DataType::kInt8)
  .Output("label", "dtype")
  .Output("tag", DataType::kInt32);
```

自定义python reader，这种方式是支持任意原始数据格式。

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('./data')
# python读取函数，直接使用tf封装好的api读取mnist数据
def read_data(batch_size=100):
    global mnist_data
    images, labels = mnist_data.train.next_batch(batch_size)
    labels = np.asarray(labels, np.float32)
    return images, labels
# 通过xdl.py_func定义op
images, labels = xdl.py_func(read_data, [], output_type=[np.float32, np.float32])
```

组网：

```python
#模型定义
model = Model_DIN(EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
#数据集
sample_io = SampleIO(train_file, test_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info, batch_size, maxlen, EMBEDDING_DIM)
with xdl.model_scope('train'):
    train_ops = model.build_final_net(EMBEDDING_DIM, sample_io)
    lr = 0.001
    # Adam Adagrad
    train_ops.append(xdl.Adam(lr).optimize())
    hooks = []
    log_format = "[%(time)s] lstep[%(lstep)s] gstep[%(gstep)s] lqps[%(lqps)s] gqps[%(gqps)s] loss[%(loss)s]"
    hooks = [QpsMetricsHook(), MetricsPrinterHook(log_format)]
    if xdl.get_task_index() == 0:
        hooks.append(xdl.CheckpointHook(xdl.get_config('checkpoint', 'save_interval')))
    train_sess = xdl.TrainSession(hooks=hooks)
with xdl.model_scope('test'):
    test_ops = model.build_final_net(
        EMBEDDING_DIM, sample_io, is_train=False)
        test_sess = xdl.TrainSession()
    model.run(train_ops, train_sess, test_ops, test_sess, test_iter=test_iter)

```


**tf_wrapper**: python decorator to adapt a tf-model define function to xdl

**组网中的tf_wrapper做了什么**

 - xdl中每个model scope对应一个tf.Graph()
 - 将使用xdl api定义的xdl dense或embedding tensor转换成tf.placeholder
 - 添加反向op，并为batch norm层添加 ps_apply_moving_average_op
 - 添加xdl.tfbackend_op，将tf的计算图作为一个op

另外，当用户定义xdl.embedding，会添加xdl.ps_sparse_pull_op和pooling
（embedding_ops.ksum/embedding_ops.kmean/embedding_ops.tile），
并且xdl.Variable中添加xdl.ps_pull_op 


**xdl.Adam(lr).optimize()做了什么**

 - 找到所有的 trainable variables
 - 从全局字典_VAR_MAPPING里拿到var对应的grad var
 - 如果var是embedding类型，相同var的梯度merge（多分支共享embedding），对每个var添加PsSparseApplyAdamOp，里面封装了ps client的SparsePush
 - 如果是dense类型，对每个var添加PsDenseApplyAdamOp，里面封装了ps client的DensePush
 - 添加一个计数op PsAssignAddOp：\_GLOBAL_STEP是个全局Variable，记录参数update的次数。


**hook**: 每次sessin.run前调用hook.before_run， sessin.run后调用hook.after_run


**current_env()**返回DistributedEnv对象，定义了所有的环境相关的变量，例如当前的task_name(schedular/server/worker)，memory_m，cpu_cores等，并根据task_name，schedular/server调用os.system启动相应的二进制。

**sessin**

xdl中定义了自己的session，session的run调用了execute方法。由于xdl支持多个后端，因此需要有一个中间表达层，即这里的graph，把通过原生api组网的网络结构，转换成graph，然后run graph里的每个node。

```python
def execute(outputs, run_option=None, run_statistic=None):
  return current_graph().execute(outputs, run_option, run_statistic)
```

```python
void Executor::Run(const GraphDef& graph,
                   const OutputSpec& output,
                   const RunOption& run_option,
                   Callback done) {
  auto graph_creator =
  [](const GraphDef& def, const OutputSpec& output, Graph** g) -> Status {
    GraphDef real_def = def;
    OutputSpec real_output = output;
    XDL_CHECK_STATUS(
        GrapplerRegistry::Get()->Process(&real_def, &real_output));
    std::unique_ptr<Graph> ret(new Graph);
    GraphBuilder builder(real_def, real_output, ret.get());
    XDL_CHECK_STATUS(builder.Build());
    *g = ret.release();
    return Status::Ok();
  };
  Graph* g;
  Status build_status =
    GraphCache::Get()->GetGraph(graph_creator, graph, output, &g);
  if (!build_status.IsOk()) {
    done(build_status, std::vector<Tensor>(), SimpleExecutor::ExtraInfo());
    return;
  }
  SimpleExecutor::Run(g, run_option, done, thread_pool_);
}
 
 
void SimpleExecutor::Run() {
  Init();
  if (!status_.IsOk()) {
    Done();
    return;
  }
  running_counter_ = 1;
  for (auto item : graph_->nodes[Graph::kSource].outputs) {
    Launch(item.node_id);
  }
  DecreaseRunningCounter();
}
```

xdl中的TFBackendOp类将xdl中的op描述转换成tf的op，由TFRunner调用tensorflow的seesion.run

```python
Status TFRunner::Run(const InputList &inputs,
                     const std::vector<std::string> &ops_names,
                     std::vector<tensorflow::Tensor>* outputs) {
  tensorflow::Status status;
  status = session_->Run(inputs, ops_names, {}, outputs);
  if (!status.ok()) {
    return Status::Internal("tf session run failed, errormsg:" +
                            status.error_message());
  }
  return Status::Ok();
}
```

每次run一个node，调用的是都需要把xdl的tensor转成tensorflow的tensor，传入自定义的alloctor避免内存拷贝

```python
Status XDL2TF::ConvertTensor(const xdl::Tensor& s, tensorflow::Tensor* d) {
  tensorflow::DataType type;
  tensorflow::TensorShape shape;
  XDL_CHECK_STATUS(ConvertType(s.Type(), &type));
  XDL_CHECK_STATUS(ConvertShape(s.Shape(), &shape));
  //avoid memcopy
  static __thread FakeAllocator fakeAlloc;
  fakeAlloc.SetBuffer(s.Raw<void>());
  *d = tensorflow::Tensor(&fakeAlloc, type, shape);
  return Status::Ok();
}
```

**sevrer端**

存参数：
 - sparse 哈希
 - dense 每一个var拆成若干part，每个part分到一个server上
 
server端多个用户态线程处理请求，并且可以设置对应的内核线程数。

**scheduler**

负责barrier、同步、worker获取训练文件、通知每个server做ServerSave／ServerRestore

**model server**

包含forward和backward方法，即前向与反向
ModelServerService负责接收请求，基于事件驱动的异步框架seastar，异步处理请求。
存参数没有分shard

**udf**

udf模块相当于pslib的accessor

**通信**

通信时候不区分sparse和dense参数。

server端异步接收请求，但是处理请求的时候，采用了读写锁。

**同步**

通过hook实现同步，有全同步SyncRunHook和半同步SemiSyncRunHook

 - 全同步：每次session.run前后都有同步
 - 半同步：每次session.run前同步（可以设置staleness），每个节点训练结束同步


client端dense pull: 每个var是切分存到各个server上的，因此pull下来之后需要combine成一个var 

 - partitioner::Broadcast var name  → partitioner::Dense 做combine



client端dense push:  

 - partitioner::Broadcast var name  → partitioner::Dense  切分梯度到各个server



client端 sparse pull:   

 - partitioner::SparseId  切分sparse id到各个server

 - partitioner::Broadcast  var name

 - partitioner::SparseData  combine


**server端参数**： 

文件build_dense_slice.cc  、build_hash_slice.cc  、build_sparse_slice.cc


sparse 参数存在一个线程安全字典中，没有分shard

[https://github.com/alibaba/x-deeplearning/blob/master/xdl/ps-plus/ps-plus/common/hashmap.h](https://github.com/alibaba/x-deeplearning/blob/master/xdl/ps-plus/ps-plus/common/hashmap.h)
