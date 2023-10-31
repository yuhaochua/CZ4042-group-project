import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import torch

train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])])

testval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=train_transform)
val_dataset = torchvision.datasets.Flowers102(root='./data', split='val', download=True, transform=testval_transform)
test_dataset = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=testval_transform)