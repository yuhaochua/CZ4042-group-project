import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np
import math

train_transform = transforms.Compose([
    transforms.RandomRotation(30), # random rotation of images
    transforms.RandomResizedCrop(224), # sample random 224x224 patch of images
    transforms.RandomHorizontalFlip(), # random horizontal flip of images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
    # transforms.Normalize([0.51796, 0.41106, 0.32971], 
    #                     [0.29697, 0.24955, 0.28531])
    ])

testval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
    # transforms.Normalize([0.51796, 0.41106, 0.32971], 
    #                     [0.29697, 0.24955, 0.28531])
])

train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=train_transform)
val_dataset = torchvision.datasets.Flowers102(root='./data', split='val', download=True, transform=testval_transform)
test_dataset = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=testval_transform)

if __name__ == "__main__":
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    sum_R = 0.0
    sum_G = 0.0
    sum_B = 0.0

    num_batches = len(train_loader) + len(val_loader) + len(test_loader) 
    num_images = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    print(num_batches)
    print(num_images)

    # MEAN 
    for batch_idx, (images, labels) in enumerate(train_loader):
        for image in images:
            numpy_image = image.numpy()
            
            sum_R += np.mean(numpy_image[0, :, :])
            sum_G += np.mean(numpy_image[1, :, :])
            sum_B += np.mean(numpy_image[2, :, :])

    for batch_idx, (images, labels) in enumerate(val_loader):
        for image in images:
            numpy_image = image.numpy()
            
            sum_R += np.mean(numpy_image[0, :, :])
            sum_G += np.mean(numpy_image[1, :, :])
            sum_B += np.mean(numpy_image[2, :, :])

    for batch_idx, (images, labels) in enumerate(test_loader):
        for image in images:
            numpy_image = image.numpy()
            
            sum_R += np.mean(numpy_image[0, :, :])
            sum_G += np.mean(numpy_image[1, :, :])
            sum_B += np.mean(numpy_image[2, :, :])
            
            
    mean_R = sum_R / num_images
    mean_G = sum_G / num_images
    mean_B = sum_B / num_images

    print("mean_R =", mean_R)
    print("mean_G =", mean_G)
    print("mean_B =", mean_B)

    variance_sum_R = 0.0
    variance_sum_G = 0.0
    variance_sum_B = 0.0

    # STD
    for batch_idx, (images, labels) in enumerate(train_loader):
        for image in images:
            numpy_image = image.numpy()
            
            variance_sum_R += np.mean(np.square(numpy_image[0, :, :] - mean_R))
            variance_sum_G += np.mean(np.square(numpy_image[1, :, :] - mean_G))
            variance_sum_B += np.mean(np.square(numpy_image[2, :, :] - mean_B))
    
    for batch_idx, (images, labels) in enumerate(val_loader):
        for image in images:
            numpy_image = image.numpy()
            
            variance_sum_R += np.mean(np.square(numpy_image[0, :, :] - mean_R))
            variance_sum_G += np.mean(np.square(numpy_image[1, :, :] - mean_G))
            variance_sum_B += np.mean(np.square(numpy_image[2, :, :] - mean_B))

    for batch_idx, (images, labels) in enumerate(test_loader):
        for image in images:
            numpy_image = image.numpy()
            
            variance_sum_R += np.mean(np.square(numpy_image[0, :, :] - mean_R))
            variance_sum_G += np.mean(np.square(numpy_image[1, :, :] - mean_G))
            variance_sum_B += np.mean(np.square(numpy_image[2, :, :] - mean_B))


    std_R = math.sqrt(variance_sum_R / num_images)
    std_G = math.sqrt(variance_sum_G / num_images)
    std_B = math.sqrt(variance_sum_B / num_images)

    print("std_R =", std_R)
    print("std_G =", std_G)
    print("std_B =", std_B)