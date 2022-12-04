# torchrun --nnodes 3 --nproc_per_node 2 --master_addr 192.255.250.15 --master_port 12345 --node_rank 0 ddp.py

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import os

local_rank = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

device = torch.device("cuda", local_rank)
model = nn.Linear(10, 10).to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for i in range(10):
    model.zero_grad()
    # 前向传播
    outputs = model(torch.randn(20, 10).to(local_rank))
    labels = torch.randn(20, 10).to(local_rank)
    loss_fn(outputs, labels).backward()
    # 后向传播
    optimizer.step()

print(list(model.parameters()))

dist.destroy_process_group()