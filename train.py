from dataset import train_dataset, val_dataset, test_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import json
import torchvision.ops as ops
import random
import numpy as np
from dcn import DeformableConv2d
from common_utils import EarlyStopper


from model import CNN, vgg, resnet, mobilenet

with open('./data/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCH = 100

# HYPERPARAMS TO TUNE
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


def eval(dataloader, model, criterion, device):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)

            correct += pred.eq(target.view_as(pred)).sum().item() # compare predicted label to actual label
    return correct / len(dataloader.dataset), total_loss / len(dataloader)
    


if __name__ == "__main__":
    torch.manual_seed(0)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    models = ['resnet', 'mobilenet']
    
    for m in models:
        if(m == 'vgg'):
            model,optimizer,criterion = vgg()
            print('Training vgg...')
        elif(m == 'resnet'):
            model,optimizer,criterion = resnet() 
            print('Training resnet...')
        elif(m == 'mobilenet'):
            model,optimizer,criterion = mobilenet()
            print('Training mobilenet...')

        model.to(DEVICE)

        early_stopper = EarlyStopper(patience=EARLY_STOP_THRESHOLD)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        best_acc = 0
        early_stop_count = 0
        for epoch in range(1, NUM_EPOCH+1):
            train_loss = train(train_loader, model, criterion, optimizer, DEVICE)
            accuracy, val_loss = eval(val_loader, model, criterion, DEVICE)
            print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Accuracy: {accuracy}, Val Loss: {val_loss}')
            if early_stopper.early_stop(val_loss):
                print("Early Stopping...") 
                torch.save(model.state_dict(), f'./saved_models/{m}_model_weights.pth')
                break
            scheduler.step()
        test_accuracy, _ = eval(test_loader, model, criterion, DEVICE)
        print(f'Test Accuracy: {test_accuracy}')

    # model,optimizer,criterion = vgg(deformable=True) # accuracy: 0.8002927....
    # model,optimizer,criterion = resnet(deformable=True) # accuracy: 0.8520084...
    # model,optimizer,criterion = mobilenet(deformable=True) # accuracy: 0.8183444...
    # model.to(DEVICE)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # best_acc = 0
    # early_stop_count = 0
    # for epoch in range(1, NUM_EPOCH+1):
    #     train_loss = train(train_loader, model, criterion, optimizer, DEVICE)
    #     accuracy, val_loss = eval(val_loader, model, criterion, DEVICE)
    #     print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Accuracy: {accuracy}')
    #     if accuracy > best_acc:
    #         best_acc = accuracy
    #         early_stop_count = 0
    #     else:
    #         early_stop_count += 1
    #     if early_stop_count >= EARLY_STOP_THRESHOLD:
    #         print("Early Stopping...")
    #         torch.save(model.state_dict(), f'./saved_models/resnet_model_weights.pth')    
    #         break
    #     scheduler.step()
    # test_accuracy, _ = eval(test_loader, model, criterion, DEVICE)
    # print(f'Test Accuracy: {test_accuracy}')
