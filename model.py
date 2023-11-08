import torch.nn as nn
import torch
import torchvision.models as models
from collections import OrderedDict
import torch.optim as optim
from dcn import DeformableConv2d

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
    
def resnet(dropout=0.5, lr=0.001, deformable=False):
    model = models.resnet152(weights='IMAGENET1K_V1')

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(OrderedDict([ #resnet152
                ('dropout1', nn.Dropout(dropout)),
                ('fc1', nn.Linear(2048, 102)),
                ('output', nn.LogSoftmax(dim=1))
                ]))
    
    # conv = deform_conv2d()

    criterion = nn.NLLLoss()

    optimizer = optim.Adam([ #resnet
        {'params': model.conv1.parameters(),  'lr': 0.000001},
        {'params': model.layer1.parameters(), 'lr': 0.000001},
        {'params': model.layer2.parameters(), 'lr': 0.00001},
        {'params': model.layer3.parameters(), 'lr': 0.00001},
        {'params': model.layer4.parameters(), 'lr': 0.0001},
        {'params': model.fc.parameters(),     'lr': 0.001}
    ], lr=0.0, weight_decay=0.001)

    if deformable: # replace last 2 convolution in resnet with deformable convolution
        model.layer4[2].conv2 = DeformableConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.layer4[2].conv3 = DeformableConv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)

    return model , optimizer ,criterion

def vgg(dropout=0.5, lr=0.001, deformable=False):
    
    model = models.vgg16(weights='DEFAULT')
        
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([ #vgg16
                        ('dropout1', nn.Dropout(dropout)),
                        ('fc2', nn.Linear(25088, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam([ #vgg16
        {'params': model.features.parameters(),  'lr': 0.00001},
        {'params': model.classifier.parameters(), 'lr': 0.001}
    ], lr=0.0, weight_decay=0.001)

    if deformable: # replace last 2 convolution in vgg16 with deformable convolution
        model.features[26] = DeformableConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.features[28] = DeformableConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    
    return model , optimizer ,criterion
    
def mobilenet(dropout=0.5, lr=0.001, deformable=False):
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([ #mobilenet
                        ('dropout1', nn.Dropout(dropout)),
                        ('fc2', nn.Linear(1280, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))
 
    criterion = nn.NLLLoss()
    optimizer = optim.Adam([ #mobilenet
        {'params': model.features.parameters(),  'lr': 0.00001},
        {'params': model.classifier.parameters(), 'lr': 0.001}
    ], lr=0.0, weight_decay=0.001)

    if deformable: # replace last 2 convolution in mobilenet with deformable convolution
        model.features[17].conv[2] = DeformableConv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        model.features[18][0] = DeformableConv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)

    return model , optimizer ,criterion