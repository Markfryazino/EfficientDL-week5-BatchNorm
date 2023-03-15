for ngpu in 1 2 3 4
do
    python -m torch.distributed.run --nproc_per_node $ngpu ddp_cifar100.py --wandb ngpu-${ngpu} --mode custom --momentum 1
done