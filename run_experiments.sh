for momentum in 0.2 0.4 0.6 0.8 1.0
do
    python -m torch.distributed.run --nproc_per_node 4 ddp_cifar100.py --wandb m-custom-${momentum} --mode custom --momentum $momentum
    python -m torch.distributed.run --nproc_per_node 4 ddp_cifar100.py --wandb m-torch-${momentum} --mode torch --momentum $momentum
done