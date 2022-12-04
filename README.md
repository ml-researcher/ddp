# ddp使用样例

## torch.distributed

https://pytorch.org/docs/stable/distributed.html

提供了一些进程之间传输数据的原语，是搭建其他高阶功能的基础。

### 点对点通信

send、recv

`dist/send.py`

### 广播通信

broadcast、broadcast_object_list、all_reduce

`dist/broad.py`

## torch.nn.parallel.DistributedDataParallel

基于torch.distributed搭建的同步神经网络梯度的类。

`ddp.py`

## DataLoader

* world_rank world_size
* worker_id num_worker

`mnist.py`

## Evaluation

TorchMetrics

上面的DataLoader部分显示了不同rank加载同样数据，只是shuffle不同。这里Evaluation显示的是每个部分加载一个数据子集。

`eval.py` & `eval_ddp.py`