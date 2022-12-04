# torchrun --nnodes 3 --nproc_per_node 2 --master_addr 192.255.250.15 --master_port 12345 --node_rank 0 mnist.py

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import os
from torchvision import datasets, transforms

# import urllib
# proxy = urllib.request.ProxyHandler({'http': 'http://127.0.0.1:7890', 'https':'http://127.0.0.1:7890'})
# opener = urllib.request.build_opener(proxy)
# urllib.request.install_opener(opener)

local_rank = int(os.environ["LOCAL_RANK"])
world_rank = int(os.environ["RANK"])
# torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

device = torch.device("cuda", local_rank)

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5,],std=[0.5,])])

if world_rank == 0: # 如果不是多个节点共享文件系统，换成local_rank == 0
    data_train = datasets.MNIST(root = "./",
                                transform=transform,
                                train = True,
                                download = True)
    dist.barrier()
else:
    dist.barrier()
    data_train = datasets.MNIST(root = "./",
                                transform=transform,
                                train = True,
                                download = False)

# data_test = datasets.MNIST(root="./",
#                            transform = transform,
#                            train = False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = 64,
                                                shuffle = True,
                                                num_workers = 4)

# data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
#                                                batch_size = 64,
#                                                shuffle = False)

for i, batch in enumerate(data_loader_train):
    print(batch)
    break

dist.destroy_process_group()