import os
import wandb

from tqdm import tqdm
from time import perf_counter

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100


from syncbn import SyncBatchNorm

torch.set_num_threads(1)


DIM = None
BS = None


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom = SyncBatchNorm(DIM, momentum=0.1)
        self.torch = nn.BatchNorm1d(DIM, momentum=0.1, affine=False)

    def forward(self, x, x2):
        res_custom = self.custom(x)
        res_torch = self.torch(x2)
        return res_custom, res_torch


def run_training(rank, size):
    torch.manual_seed(0)

    model = Net()

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    model.to(device)

    mae_forward = mae_backward = 0
    mse_forward = mse_backward = 0

    if rank == 0:
        pbar = tqdm(total=1000)
    for i in range(1000):
        x = torch.rand((BS, DIM)).to(device)
        x.requires_grad = True
        x2 = x.clone().detach()
        x2.requires_grad = True

        res_custom, res_torch = model(x, x2)
        rc = res_custom[: BS // 2].sum()
        rc.backward()
        rt = res_torch[: BS // 2].sum()
        rt.backward()
        grad_custom = x.grad
        grad_torch = x2.grad

        mae_forward += torch.abs(res_custom - res_torch).mean()
        mae_backward += torch.abs(grad_custom - grad_torch).mean()
        mse_forward += ((res_custom - res_torch) ** 2).mean()
        mse_backward += ((grad_custom - grad_torch) ** 2).mean()
        if rank == 0:
            pbar.update(1)

    wandb.log({
        "rank": rank,
        "mae_forward": mae_forward / 1000,
        "mae_backward": mae_backward / 1000,
        "mse_forward": mse_forward / 1000,
        "mse_backward": mse_backward / 1000
    })

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", type=str, default=None)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--bs", type=int, default=32)

    return parser.parse_args()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])

    args = parse_args()
    DIM = args.dim
    BS = args.bs

    if local_rank == 0:
        wandb.init(
            entity="broccoliman",
            project="efficient_dl_week5",
            name=args.wandb,
            config={
                "dim": DIM,
                "batch_size": BS
            }
        )
    

    init_process(local_rank, fn=run_training, backend="nccl")  # replace with "nccl" when testing on GPUs
