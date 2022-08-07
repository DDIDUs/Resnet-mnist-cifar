from random import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as model

import numpy as np

if __name__ == "__main__":
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    print(train_loader)
    for i, j in enumerate(train_loader):
        print(i, j[0].shape, j[1].shape)