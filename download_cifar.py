from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


dataset = CIFAR100(
    "./cifar",
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    ),
    download=True,
)

val_dataset = CIFAR100(
    "./cifar",
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    ),
    download=True,
    train=False
)