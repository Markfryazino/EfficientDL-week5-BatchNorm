python -m torch.distributed.run --nproc_per_node 1 synthetic_exp.py --wandb last-syn-1-128-32 --dim 128 --bs 32
python -m torch.distributed.run --nproc_per_node 1 synthetic_exp.py --wandb last-syn-1-128-64 --dim 128 --bs 64
python -m torch.distributed.run --nproc_per_node 1 synthetic_exp.py --wandb last-syn-1-256-32 --dim 256 --bs 32
python -m torch.distributed.run --nproc_per_node 1 synthetic_exp.py --wandb last-syn-1-256-64 --dim 256 --bs 64
python -m torch.distributed.run --nproc_per_node 4 synthetic_exp.py --wandb last-syn-4-128-32 --dim 128 --bs 32
python -m torch.distributed.run --nproc_per_node 4 synthetic_exp.py --wandb last-syn-4-128-64 --dim 128 --bs 64
python -m torch.distributed.run --nproc_per_node 4 synthetic_exp.py --wandb last-syn-4-256-32 --dim 256 --bs 32
python -m torch.distributed.run --nproc_per_node 4 synthetic_exp.py --wandb last-syn-4-256-64 --dim 256 --bs 64