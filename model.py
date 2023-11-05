import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.ops import deform_conv2d
from collections import OrderedDict
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding='same')  # Same padding
        self.p1 = nn.MaxPool2d(2, 2)  # Max pooling over a (2, 2) window
        self.conv2 = nn.Conv2d(32, 64, 5, padding='same')  # Same padding
        self.p2 = nn.MaxPool2d(2, 2)  # Max pooling over a (2, 2) window
        self.flatten = nn.Flatten()
        self.d1 = nn.Linear(56 * 56 * 64, 1024)
        self.dropout = nn.Dropout(0.2)
        self.d2 = nn.Linear(1024, 102)

    def forward(self, x):
        conv1 = torch.relu(self.conv1(x))
        p1 = self.p1(conv1)
        conv2 = torch.relu(self.conv2(p1))
        p2 = self.p2(conv2)
        flatten = self.flatten(p2)
        d1 = torch.relu(self.d1(flatten))
        dropout = self.dropout(d1)
        out = torch.softmax(self.d2(dropout), dim=1)
        return conv1, p1, conv2, p2, out
    
def resnet(dropout=0.5):
    model = models.resnet152(pretrained=True)
    model_requires_grad_params = []

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    # conv = deform_conv2d()

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(
    [
        {'params': model.conv1.parameters(),  'lr': 0.000001},
        {'params': model.layer1.parameters(), 'lr': 0.000001},
        {'params': model.layer2.parameters(), 'lr': 0.00001},
        {'params': model.layer3.parameters(), 'lr': 0.00001},
        {'params': model.layer4.parameters(), 'lr': 0.0001},
        {'params': model.fc.parameters(),     'lr': 0.001}
    ], lr=0.0, weight_decay=0.001)

    return model , optimizer ,criterion

def vgg(dropout=0.5, lr = 0.001):
    
    model = models.vgg16(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                        ('dropout1', nn.Dropout(dropout)),
                        ('fc2', nn.Linear(25088, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    
    return model , optimizer ,criterion
    
def mobilenet(dropout=0.5, lr = 0.001):
    model = models.mobilenet_v2(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                        ('dropout1', nn.Dropout(dropout)),
                        ('fc2', nn.Linear(1280, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))
    
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return model , optimizer ,criterion