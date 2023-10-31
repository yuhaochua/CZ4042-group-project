import torch.nn as nn
import torch

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