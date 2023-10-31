from dataset import train_dataset, val_dataset, test_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import json
import torchvision.models as models

from model import CNN

with open('./data/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCH = 100
NUM_CLASSES = 5

# HYPERPARAMS TO TUNE
NUM_HIDDEN = 128
NUM_LAYERS = 1
BATCH_SIZE = 128
EARLY_STOP_THRESHOLD = 3
LR = 0.001

def train(dataloader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    return total_loss/len(dataloader)


def eval(dataloader, model, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1)

            correct += pred.eq(target.view_as(pred)).sum().item() # compare predicted label to actual label
    return correct / len(dataloader.dataset)

def nn_setup(dropout=0.5, hidden_layer1 = 120,lr = 0.001):
    
    model = models.vgg16(pretrained=True)  
        
    for param in model.parameters():
        param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 500)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )
        
        
        return model , optimizer ,criterion

if __name__ == "__main__":
    loss_list = []
    accuracy_list = []
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # model = CNN()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    model,optimizer,criterion = nn_setup()
    model.to(DEVICE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_acc = 0
    early_stop_count = 0
    for epoch in range(1, NUM_EPOCH+1):
        train_loss = train(train_loader, model, criterion, optimizer, DEVICE)
        accuracy = eval(val_loader, model, DEVICE)
        print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Accuracy: {accuracy}')
        if accuracy > best_acc:
            best_acc = accuracy
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= EARLY_STOP_THRESHOLD:
            print("Early Stopping...")        
            break
        scheduler.step()
    test_accuracy = eval(test_loader, model, DEVICE)
    print(f'Test Accuracy: {test_accuracy}')
