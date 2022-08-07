from random import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as model

import numpy as np

def asdf(train_dataset,sp):
    if sp == "mnist":
        label1 = [0,1,2,3,4,5,6,7,8,9]
        t1 = []
        for i in label1:
            t = []
            for j in train_dataset:
                if j[1] == i:
                    t.append(j)
            t1.append(torch.utils.data.DataLoader(tuple(t), batch_size=256, shuffle=False, num_workers=8))
    else:
        label2 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        t1 = []
        for i in label2:
            t = []
            for j in train_dataset:
                if j[1] == i:
                    t.append(j)
            t1.append(torch.utils.data.DataLoader(t, batch_size=256, shuffle=False, num_workers=8))
    return t1


def Load_Cifar10(sp=None):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_dataset, vaild_dataset = torch.utils.data.random_split(dataset, [45000, 5000])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    if sp == 1:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    elif sp ==2:
        train_loader = asdf(train_dataset, "cifar10")
    else:
        train_loader = torch.utils.data.DataLoader(tuple(sorted(train_dataset, key=lambda x: x[1])), batch_size=256, shuffle=True, num_workers=8)
    vaild_loader = torch.utils.data.DataLoader(vaild_dataset, batch_size=64, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=8)
    return train_loader, vaild_loader, test_loader



def Load_MNIST(sp = None):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    train_dataset, vaild_dataset = torch.utils.data.random_split(dataset, [50000, 10000])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    if sp == 1:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    elif sp == 2:
        train_loader = asdf(train_dataset, "mnist")
    else:
        train_loader = torch.utils.data.DataLoader(tuple(sorted(train_dataset, key=lambda x: x[1])), batch_size=256, shuffle=True, num_workers=8)
    vaild_loader = torch.utils.data.DataLoader(vaild_dataset, batch_size=256, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=8)
    return train_loader, vaild_loader, test_loader