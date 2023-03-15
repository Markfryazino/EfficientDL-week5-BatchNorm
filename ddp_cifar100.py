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


MOMENTUM = None
GAS = None
MODE = None


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        # self.bn1 = nn.BatchNorm1d(128, affine=False)  # to be replaced with SyncBatchNorm

        if MODE == "custom":
            self.bn1 = SyncBatchNorm(128, momentum=MOMENTUM)
        else:
            self.bn1 = nn.SyncBatchNorm(128, momentum=MOMENTUM, affine=False)

    def forward(self, x, run_all_reduce=True):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        
        kwargs = {"run_all_reduce": run_all_reduce} if MODE == "custom" else {}
        x = self.bn1(x, **kwargs)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def run_training(rank, size):
    torch.manual_seed(0)

    dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=False,
    )

    val_dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=False,
        train=False
    )

    # where's the validation dataset?
    loader = DataLoader(dataset, sampler=DistributedSampler(dataset, size, rank), batch_size=64)
    val_loader = DataLoader(val_dataset, sampler=DistributedSampler(val_dataset, size, rank), batch_size=64)

    model = Net()

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if MODE != "custom":
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)


    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_steps = 0
    logging_loss = forwards = backwards = 0

    if local_rank == 0:
        pbar = tqdm(total=len(loader) * 10)

    for epoch in range(10):
        model.train()
        for data, target in loader:
            num_steps += 1
            data = data.to(device)
            target = target.to(device)

            forwards -= perf_counter()
            output = model(data, run_all_reduce=num_steps % GAS == 0)
            forwards += perf_counter()
            loss = torch.nn.functional.cross_entropy(output, target)
            backwards -= perf_counter()
            loss.backward()
            backwards += perf_counter()
            logging_loss += loss.item()

            if num_steps % GAS == 0:
                average_gradients(model)
                optimizer.step()
                optimizer.zero_grad()

            if num_steps % 20 == 0 and local_rank == 0:
                wandb.log({"loss": logging_loss / 20, "epoch": epoch}, step=num_steps)
                if num_steps > 20:
                    wandb.log({"average_forward_time": forwards / 20, "average_backward_time": backwards / 20}, step=num_steps)
                logging_loss = forwards = backwards = 0
            
            if local_rank == 0:
                pbar.update(1)
        
        num_guesses = val_loss = num_samples = 0
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data.to(device), run_all_reduce=num_steps % GAS == 0)
                loss = torch.nn.functional.cross_entropy(output, target.to(device))
                val_loss += loss.item()
                num_guesses += (output.argmax(dim=1) == target.to(device)).float().sum()
                num_samples += data.size(0)

        metrics = torch.tensor([num_guesses, val_loss, 
            torch.tensor(num_samples), torch.tensor(len(val_loader))]).to(device)
        dist.all_reduce(metrics)
        
        if local_rank == 0:
            wandb.log({
                "val_loss": (metrics[1] / metrics[3]).item(), 
                "val_accuracy": (metrics[0] / metrics[2]).item(), 
                "epoch": epoch
                },
                step=num_steps
            )



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", type=str, default=None)
    parser.add_argument("--momentum", type=float, default=0.1)
    parser.add_argument("--gas", type=int, default=1)
    parser.add_argument("--mode", type=str, default="custom")

    return parser.parse_args()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])

    args = parse_args()
    MOMENTUM = args.momentum
    GAS = args.gas
    MODE = args.mode

    if local_rank == 0:
        wandb.init(
            entity="broccoliman",
            project="efficient_dl_week5",
            name=args.wandb,
            config={
                "momentum": MOMENTUM,
                "gas": GAS,
                "mode": MODE
            }
        )
    

    init_process(local_rank, fn=run_training, backend="nccl")  # replace with "nccl" when testing on GPUs
