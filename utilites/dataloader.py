import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

from config import BATCH_SIZE, DATA_PATH, NUM_WORKERS


def get_data_loaders():
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter()], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load the full training set
    full_trainset = datasets.CIFAR10(
        root=DATA_PATH, train=True, download=True, transform=transform_train
    )

    train_val_split_factor = 0.8  # 80%
    train_size = int(train_val_split_factor * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    testset = datasets.CIFAR10(
        root=DATA_PATH, train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader
