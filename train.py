from dataset import train_dataset, val_dataset, test_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import math


from model import CNN, vgg, resnet, mobilenet

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
    


if __name__ == "__main__":
    loss_list = []
    accuracy_list = []
    torch.manual_seed(0)
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    # model,optimizer,criterion = vgg() # accuracy: 0.8002927....
    # model,optimizer,criterion = resnet() # accuracy: 0.8520084...
    model,optimizer,criterion = mobilenet() # accuracy: 0.8183444...
    model.to(DEVICE)

    print(model.features[0])

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # best_acc = 0
    # early_stop_count = 0
    # for epoch in range(1, NUM_EPOCH+1):
    #     train_loss = train(train_loader, model, criterion, optimizer, DEVICE)
    #     accuracy = eval(val_loader, model, DEVICE)
    #     print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Accuracy: {accuracy}')
    #     if accuracy > best_acc:
    #         best_acc = accuracy
    #         early_stop_count = 0
    #     else:
    #         early_stop_count += 1
    #     if early_stop_count >= EARLY_STOP_THRESHOLD:
    #         print("Early Stopping...")        
    #         break
    #     scheduler.step()
    # test_accuracy = eval(test_loader, model, DEVICE)
    # print(f'Test Accuracy: {test_accuracy}')
