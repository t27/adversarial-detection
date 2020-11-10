# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:51:26 2020

@author: vidhi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class AdvDetector(nn.Module):
    def __init__(self, in_channels, pooling1=False, pooling2=False):
        super(AdvDetector, self).__init__()
        
        self.pooling1=pooling1
        self.pooling2=pooling2
        
        self.conv1=nn.Conv2d(in_channels=in_channels, out_channels=96,kernel_size=3,stride=1, padding=1)
        self.layer2=nn.BatchNorm2d(96)
        self.layer3=nn.ReLU()
        self.layer4=nn.MaxPool2d(2,2) #1st pooling layer
        self.layer5=nn.Conv2d(in_channels=96, out_channels=192,kernel_size=3,stride=1,padding=1)
        self.layer6=nn.BatchNorm2d(192)
        self.layer7=nn.ReLU()
        self.layer8=nn.MaxPool2d(2,2)  #2nd pooling layer
        self.layer9=nn.Conv2d(in_channels=192, out_channels=192,kernel_size=3,stride=1, padding=1)
        self.layer10=nn.BatchNorm2d(192)
        self.layer11=nn.ReLU()
        self.layer12=nn.Conv2d(in_channels=192,out_channels=2,kernel_size=1,stride=1)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self,x):        
        x = self.conv1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.pooling1==True:
            x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        if self.pooling2==True:
            x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        
        x = self.adaptive_avg_pool(x)
        return x

class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)


def accuracy(y_pred, y_test):
    y_pred_tag = torch.argmax(torch.round(torch.sigmoid(y_pred)))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def train(train_loader, model, criterion, optimizer, epoch, loss_function):
    device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "cpu")
    device="cpu"
    model.train()
    for e in range(1, epoch+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
        
            y_pred = model(X_batch)
            y_pred=y_pred.reshape(-1,2)
            #print(y_pred.reshape(-1,2).shape)
            #print(y_batch.shape)
            loss = criterion(y_pred, y_batch.long())
            acc = accuracy(y_pred, y_batch.long())
        
            loss.backward()
            optimizer.step()
        
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        

        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
        
def test(test_loader, model, criterion, optimizer, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "cpu")
    device="cpu"
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
        
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    print(y_pred_list)
    
def main():
    Epoch=20
    Batch_size=4
    lr=0.001
    #file=np.load('c1s_train.npy')
    device = "cpu"
    
    #input
    temp_data = np.random.randn(128, 64, 8, 8)
    temp_data = torch.Tensor(temp_data)
    X_train=temp_data #train_data
    X_test=temp_data #test_data
    
    # labels
    target = torch.ones((128))
    y_train=target
    
    
    train_data = TrainData(torch.FloatTensor(X_train), 
                       torch.FloatTensor(y_train))
    test_data = TestData(torch.FloatTensor(X_test))
    
    train_loader = DataLoader(dataset=train_data,batch_size=Batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data,batch_size=Batch_size)
    
    model = AdvDetector(64,False,False) #model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    loss_function = nn.MSELoss()
    
    train(train_loader, model, criterion, optimizer,Epoch,loss_function)
    test(test_loader, model, criterion, optimizer,Epoch)
    
    
    
if __name__ == "__main__":
    main()