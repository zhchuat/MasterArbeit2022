from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd

torch.manual_seed(1)  # reproducible

"""Hyper Parameter"""
batchsize = 2
learning_rate = 0.00001
#set total epoch
totalepoch = 5

transform_train = transforms.Compose([  
                                        # transforms.RandomCrop(480),
                                        # transforms.Resize(640),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
                                        ])

transform_val = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5,0.5), (0.5, 0.5, 0.5, 0.5))
                                        ])

'''creat train dataset'''
trainRGBD = np.load('/home/zhu/dataset/RGBD.npy')
train_Labels = np.load('/home/zhu/dataset/labels.npy')

class train_Dataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, trainRGBD, train_Labels):
        self.data = trainRGBD
        self.label = train_Labels
        # self.transforms = transform_train
    #返回数据集大小
    def __len__(self):
        return len(self.data)
    #得到数据内容和标签
    def __getitem__(self, index):
        RGBD = torch.Tensor( self.data[index])
        RGBDLabels = torch.IntTensor(self.label[index])
        return RGBD, RGBDLabels
 
'''creat eval dataset'''

evalRGBD = np.load('/home/zhu/dataset/RGBD_eval.npy')
eval_Labels = np.load('/home/zhu/dataset/labels_eval.npy')

class eval_Dataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, evalRGBD, eval_Labels):
        self.data = evalRGBD
        self.label = eval_Labels
        # self.transforms = transform_val
    #返回数据集大小
    def __len__(self):
        return len(self.data)
    #得到数据内容和标签
    def __getitem__(self, index):
        RGBD = torch.Tensor(self.data[index])
        RGBDLabels = torch.IntTensor(self.label[index])
        return RGBD, RGBDLabels
 

'''CNN Modell'''
in_dim = len(trainRGBD[1]) #
from Model.ResNet import Conv1, Bottleneck

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=894, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 4, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d((20,15), stride=1)
        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



    # out_dir = "./checkpoints/"
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    '''Training function'''    
    def train_model(self,device):
        
        #creat optimizerimages
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #creat loss function
        #loss_func = nn.BCEWithLogitsLoss()
        loss_func = nn.CrossEntropyLoss()
        #creat tensorboard SummaryWriter
        writer = SummaryWriter("./logs_train")
        
        for epoch in range(totalepoch):  # loop over the dataset multiple times
            #start record time
            timestart = time.time()
            print("----------the {} train beginning----------".format(epoch+1))
            #initial parameter
            
            total_train_step = 0
            correct = 0.0
            total = 0.0
            val_acc_list = []

            for i, item in enumerate(trainloader):# get the inputs; data is a list of [inputs, labels]
                inputs, labels = item
                #print('i:', i)
                # print('Train image size:', inputs.shape)
                # print('Train label size:', labels.shape)
                inputs, labels = inputs.to(device), labels.cpu()  #using CUDA
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                #output [batchsize,640,480] --> [batch size, 894]
                labels = np.bincount(labels.flatten(),minlength= 894)

                labels = torch.from_numpy(labels).to(device)/640/480  
                # print('labels',labels.shape)
                # forward + backward + optimize
                output= mymodel(inputs).to(device)

                # print('output',output.shape)
                _, predicted = torch.max(output.data, dim=0)
                
                # predicted = torch.from_numpy(predicted).float().to(device)/255
                #print(predicted.shape)
                
                loss=loss_func(predicted.float() , labels.float())
                train_loss = loss.item()
                loss.requires_grad = True
                loss.backward()

                total_train_step +=1
                
                #optimizer model
                optimizer.step()
            
                total += labels.size(0)
                # print(total.shape)
                correct += (predicted == labels).sum().item()
                # print(correct.shape)
                train_acc = 100. * correct / total
                if total_train_step % 10 == 0:
                    print('---Epoch: {}----total train step is: {}, Loss: {}, Train Acc = {}%'.format(epoch+1, total_train_step,train_loss, train_acc))
                    writer.add_scalar("train_loss", train_loss, total_train_step)
                    writer.add_scalar("train_acc", train_acc, total_train_step)   

            
            
            val_acc_list.append(train_acc)
            if train_acc > max(val_acc_list):
                torch.save(mymodel.state_dict(),"best.pt")
                print("------------save a best Model--Best ACC:{}------------".format(max(val_acc_list)))
            print("Epoch: {} Cost: {} Time: sec {}".format(epoch+1, train_loss, time.time()-timestart))

        print('Finished Training')
        torch.save(mymodel.state_dict(),"last.pt")
        print("------------save a last Model------------")
        writer.close()
       
    """ Eval function""" 
    def model_eval(self,device):
        print(' Evaluation Beginning: \n')
        correct_predict = 0
        total_predict = 0
                   
        with torch.no_grad():
            for i, item in enumerate(evalloader):
                eval_img, eval_Labels = item
                eval_img, eval_Labels = eval_img.to(device), eval_Labels.cpu()
                outputs = mymodel(eval_img).to(device)
                # label image bearbeiten
                
                eval_Labels = np.bincount(eval_Labels.flatten(),minlength= 894)

                eval_Labels = torch.from_numpy(eval_Labels).float().to(device)/640/480 #output [batchsize,640,480] --> [batch size, 894]

                _, predicted = torch.max(outputs.data, dim=0)
                                
                total_predict += eval_Labels.size(0)
                correct_predict += (predicted == eval_Labels).sum().item()
                
        print('Accuracy of the network on the test images: {}'.format(
                100.0 * correct_predict / total_predict))

if __name__ == '__main__':
    #creat train_DataLoader
    train_dataset = train_Dataset(trainRGBD, train_Labels)
    trainloader = DataLoader.DataLoader(train_dataset, batch_size= batchsize, shuffle = True, num_workers= 4)
    #creat eval_DataLoader
    eval_dataset = eval_Dataset(evalRGBD, eval_Labels)
    evalloader = DataLoader.DataLoader(eval_dataset, batch_size= batchsize, shuffle = False, num_workers= 4)
    # print('the number of evaluation dataset：', evalRGBD.__len__())

    device = torch.device("cuda:0")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def ResNet50():
        return ResNet([3, 4, 6, 3])

    def ResNet101():
        return ResNet([3, 4, 23, 3])

    def ResNet152():
        return ResNet([3, 8, 36, 3])
    
    mymodel = ResNet152()   
    # mymodel = MyCNN(in_dim, n_class=894)
    # result=mymodel.forward(train_dataset)
    mymodel = mymodel.to(device)
    mymodel.train_model(device)
    mymodel.model_eval(device)
    # torch.save(mymodel.state_dict(),'trainmodel.pt') #save model
