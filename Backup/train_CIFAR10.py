import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter

transform = transforms.Compose(
    [
     #transforms.RandomHorizontalFlip(),
     #transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform1 = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./dataset/', train=True,
                                          download=False, transform=transform)
print(type(trainset))
testset = torchvision.datasets.CIFAR10(root='./dataset/', train=False,
                                        download=False, transform=transform1)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                           shuffle=True, num_workers=0)
print(type(trainloader))
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,3,1,1),    # 1st conv，in_channels:3，out_channels:32，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
            nn.MaxPool2d(2),          # 1st Max Pooling
            nn.BatchNorm2d(32),
            nn.SiLU(),

            nn.Conv2d(32,64,3,1,1),   # 2nd conv，in_channels:32，out_channels:32，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
            nn.MaxPool2d(2),          # 2nd Max Pooling
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.Conv2d(64,128,3,1,1),  # 3rd conv，in_channels:32，out_channels:64，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
            nn.MaxPool2d(2),          # 3rd Max Pooling
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.Flatten(),
            nn.Linear(4*4*128,64),    # 1st fc with linear，in_features:4×4×64，out_features:128
            nn.Linear(64, 10),        # 2nd fc with linear，in_features:64，out_features:10
        )

    def forward(self,x):              # forward function to get the output of CNN, reference
        self.model(x)
        return self.model(x)


    def train_model(self,device):
        #creat optimizer
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        #initial epoch parameter
        initepoch = 1
        #creat loss function
        loss = nn.CrossEntropyLoss()

        #creat tensorboard SummaryWriter
        writer = SummaryWriter("./logs_train")

        for epoch in range(initepoch):  # loop over the dataset multiple times
            #start record time
            timestart = time.time()
            print("----------the {} train beginning-------------".format(epoch+1))
            #initial parameter
            running_loss = 0.0
            total_train_step = 0
            
            for data in trainloader:
                imgs, labels = data
                imgs, labels = imgs.to(device),labels.to(device)  #using CUDA
                #print(imgs.shape)
                outputs = self(imgs)
                l=loss(outputs, labels)
                train_loss = l.item()
                #optimizer model
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                total_train_step +=1
                if total_train_step % 100 == 0:
                    print('total train step is: {}, Loss: {}'.format(total_train_step,train_loss))
                    writer.add_scalar("train_loss", train_loss, total_train_step)

            print("Epoch: {} Cost: {} Time: sec {}".format(epoch+1, train_loss, time.time()-timestart))
        print('Finished Training')
        writer.close()

    def model_eval(self,device):
        correct_predict = 0
        total_predict = 0
        
        
        with torch.no_grad():
            for data in testloader:
                imgs, labels = data
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = self(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total_predict += labels.size(0)
                correct_predict += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: {}'.format(
                100.0 * correct_predict / total_predict))
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mymodel = CNN()
mymodel = mymodel.to(device)
mymodel.train_model(device)
mymodel.model_eval(device)
torch.save(mymodel.state_dict(), 'mymodel.pt') #save model