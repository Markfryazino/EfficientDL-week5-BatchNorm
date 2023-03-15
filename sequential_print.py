import os

import torch.distributed as dist


def run_sequential(rank, size, num_iter=10):
    """
    Prints the process rank sequentially according to its number over `num_iter` iterations,
    separating the output for each iteration by `---`
    Example (3 processes, num_iter=2):
    ```
    Process 0
    Process 1
    Process 2
    ---
    Process 0
    Process 1
    Process 2
    ```
    """

    for iter in range(num_iter):
        for i in range(rank):
            dist.barrier()
        
        print(f"Process {rank}", flush=True)

        for i in range(rank, size):
            dist.barrier()

        if rank == 0 and iter != num_iter - 1:
            print("---")
        dist.barrier()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(rank=local_rank, backend="gloo")

    run_sequential(local_rank, dist.get_world_size())
