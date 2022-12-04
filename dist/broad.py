# torchrun --nnodes 3 --nproc_per_node 2 --master_addr 192.255.250.15 --master_port 12345 --node_rank 0 dist.py

import os

import torch
import torch.distributed as dist

torch.manual_seed(0)

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

def run():
    tensor = torch.randn(5)
    
    device = torch.device("cuda:{}".format(LOCAL_RANK))
    tensor = tensor.to(device)

    print(WORLD_RANK, "tensor before broadcast", tensor)
    dist.broadcast(tensor, 0)
    print(WORLD_RANK, "tensor after broadcast", tensor)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(WORLD_RANK, "tensor after sum", tensor)

def init_processes():
    dist.init_process_group('nccl')
    run()
    dist.destroy_process_group()

if __name__ == "__main__":
    init_processes()