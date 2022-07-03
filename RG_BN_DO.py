# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 23:03:34 2022

@author: JCH
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


""" Dataset """

trainset = datasets.MNIST(root ="C:/Users/JCH/Python/data/MNIST",
                               train = True,
                               download = True,
                               transform = transforms.ToTensor( ))

testset = datasets.MNIST(root ="C:/Users/JCH/Python/data/MNIST",
                               train = False,
                               download = True,
                               transform = transforms.ToTensor( ))

 
EPOCHS = 10
BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    drop_last = True
)  # [len(train_loader), ([images],[labels])]

test_loader = torch.utils.data.DataLoader(
    dataset     = testset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    drop_last = False
)


""" Model """
class Net(nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        # Weight initialization
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=1.0)
        # 드롭아웃 확률
        self.dropout_p = dropout_p

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        # 드롭아웃 추가
        x = F.dropout(x, p=self.dropout_p)
        x = F.relu(self.fc2(x))
        # 드롭아웃 추가
        x = F.dropout(x, p=self.dropout_p)
        x = self.fc3(x)
        return x


model = Net(dropout_p=0.2).to(DEVICE)

""" Optimizer """
# Weight_decay : Regularization 추가
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)


""" Loss ftn """
Loss_ftn = torch.nn.CrossEntropyLoss(reduction='sum')


""" Training """
### For 1 epoch ###
def train(model, train_loader, optimizer):

    for sample in train_loader:
        image, label = sample
        image, label = image.to(DEVICE), label.to(DEVICE)

        score = model(image)
        loss = Loss_ftn(score, label)/len(image)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



""" Test """
### For 1 epoch ###
def evaluate(model, test_loader):
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
                    
            image, label = image.to(DEVICE), label.to(DEVICE)

            score = model(image)
            test_loss += Loss_ftn(score, label)

            pred = torch.argmax(score,1)
            correct += pred.eq(label.view_as(pred)).sum()


    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset) * 100.

    return test_loss, test_accuracy

 


for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader)
    
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
          epoch, test_loss, test_accuracy))


