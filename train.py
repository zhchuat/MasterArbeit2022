from PIL import Image
import skimage
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

transform_train = transforms.Compose([  
                                        transforms.RandomCrop(480),
                                        #transforms.Resize(640),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
                                        ])

transform_val = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5,0.5), (0.5, 0.5, 0.5, 0.5))
                                        ])

'''
creat train dataset
'''
trainRGBD = np.load('/home/hczhu/CNNlearn/dataset/RGBD.npy')
train_Labels = np.load('/home/hczhu/CNNlearn/dataset/labels.npy')

class train_Dataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, trainRGBD, train_Labels):
        self.data = trainRGBD
        self.label = train_Labels
        #self.transforms = transform_train
    #返回数据集大小
    def __len__(self):
        return len(self.data)
    #得到数据内容和标签
    def __getitem__(self, index):
        RGBD = torch.Tensor(self.data[index])
        RGBDLabels = torch.IntTensor(self.label[index])
        return RGBD, RGBDLabels
 
'''
creat eval dataset
'''
evalRGBD = np.load('/home/hczhu/CNNlearn/dataset/RGBD_eval.npy')
eval_Labels = np.load('/home/hczhu/CNNlearn/dataset/labels_eval.npy')

class eval_Dataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, evalRGBD, eval_Labels):
        self.data = evalRGBD
        self.label = eval_Labels
        #self.transforms = transform_val
    #返回数据集大小
    def __len__(self):
        return len(self.data)
    #得到数据内容和标签
    def __getitem__(self, index):
        RGBD = torch.Tensor(self.data[index,:,:,:])
        RGBDLabels = torch.IntTensor(self.label[index,:,:])
        return RGBD, RGBDLabels
 
# if __name__ == '__main__':
#     evalRGBD = train_Dataset(evalRGBD, eval_Labels)
#     print('the number of evaluation dataset：', evalRGBD.__len__())
#     #creat DataLoader
#     evalloader = DataLoader.DataLoader(evalRGBD,batch_size= 2, shuffle = False, num_workers= 4)

'''CNN Modell'''

in_dim = len(trainRGBD[1]) #


class MyCNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(MyCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_dim,32,3,1,1),      # 1st conv，in_channels:4，out_channels:32，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
            nn.MaxPool2d(2),            # 1st Max Pooling
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Dropout(),         
            
            nn.Conv2d(32,64,3,1,1),     # 2nd conv，in_channels:32，out_channels:64，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
            nn.MaxPool2d(2),            # 2nd Max Pooling
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Dropout(),
            
            nn.Conv2d(64,128,3,1,1),    # 3rd conv，in_channels:64，out_channels:128，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
            nn.MaxPool2d(2),            # 3rd Max Pooling
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Dropout(),
            
            nn.Flatten(),
            nn.Linear(80*60*128,64),    # 1st fc with linear，in_features:80×80×128，out_features:64
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, n_class),          # 2nd fc with linear，in_features:64，out_features:10
            nn.Softmax(dim=1)
        )

    def forward(self,x):                # forward function to get the output of CNN, reference
        out = self.model(x)
        return out

    '''Training function'''    
    def train_model(self,device):
                
        #creat optimizerimages
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        #initial epoch parameter
        initepoch = 1
        
        #creat loss function
        
        loss_func = nn.CrossEntropyLoss()

        #creat tensorboard SummaryWriter
        writer = SummaryWriter("./logs_train")

        for epoch in range(initepoch):  # loop over the dataset multiple times
            #start record time
            timestart = time.time()
            print("----------the {} train beginning----------".format(epoch+1))
            #initial parameter
            running_loss = 0.0
            total_train_step = 0
            
            for i, item in enumerate(trainloader):# get the inputs; data is a list of [inputs, labels]
                inputs, labels = item
                print('i:', i)
                # print('image:', inputs.shape)
                # print('label:', labels.shape)
                inputs, labels = inputs.to(device), labels.to(device)  #using CUDA
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # label image bearbeiten
                label_variable = nn.Sequential(
                    nn.Conv2d(1,32,3,1,1),      # 1st conv，in_channels:4，out_channels:32，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
                    nn.MaxPool2d(2),            # 1st Max Pooling
                    nn.BatchNorm2d(32),
                    nn.SiLU(),
                    nn.Dropout(),         
                    
                    nn.Conv2d(32,64,3,1,1),     # 2nd conv，in_channels:32，out_channels:64，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
                    nn.MaxPool2d(2),            # 2nd Max Pooling
                    nn.BatchNorm2d(64),
                    nn.SiLU(),
                    nn.Dropout(),
                    
                    nn.Conv2d(64,128,3,1,1),    # 3rd conv，in_channels:64，out_channels:128，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
                    nn.MaxPool2d(2),            # 3rd Max Pooling
                    nn.BatchNorm2d(128),
                    nn.SiLU(),
                    nn.Dropout(),
                    
                    nn.Flatten(),
                    nn.Linear(80*60*128,64),    # 1st fc with linear，in_features:80×80×128，out_features:64
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(64, 894),          # 2nd fc with linear，in_features:64，out_features:10
                    nn.Softmax(dim=1)
                )
                labels = torch.unsqueeze(labels,0)  #von 1,640,480 ------> 1,1,640,480 passt Modell            
                # labels = torch.IntTensor(labels)
                # print(labels.shape)
                labels = label_variable(labels.float())  #output now is 1,894
                # print(labels.shape)
                # forward + backward + optimize
                                
                output= self.model(inputs)
                # print(output.shape)

                loss=loss_func(output, labels)
                train_loss = loss.item()
                loss.backward()

                #optimizer model
                optimizer.step()
                
                total_train_step +=1
                running_loss += train_loss

                if total_train_step % 50 == 0:
                    print('total train step is: {}, Loss: {}'.format(total_train_step,train_loss))
                    writer.add_scalar("train_loss", train_loss, total_train_step)   

            print("Epoch: {} Cost: {} Time: sec {}".format(epoch+1, train_loss, time.time()-timestart))
        print('Finished Training')
        writer.close()

    # Eval function train -> eval
    # def model_eval(self,device):
    #     correct_predict = 0
    #     total_predict = 0
                
    #     with torch.no_grad():
    #         for i, item in enumerate(evalloader):
    #             eval_img, eval_Labels = item
    #             eval_img, eval_Labels = eval_img.to(device), eval_Labels.to(device)
    #             outputs = self.model(eval_img)
    #             _, predicted = torch.max(outputs.data, 1)
                
                
    #             total_predict += eval_Labels.size(0)
    #             correct_predict += (predicted == eval_Labels).sum().item()

    #     print('Accuracy of the network on the test images: {}'.format(
    #             100.0 * correct_predict / total_predict))
        

if __name__ == '__main__':
    train_dataset = train_Dataset(trainRGBD, train_Labels)
    #创建DataLoader迭代器
    trainloader = DataLoader.DataLoader(train_dataset, batch_size= 1, shuffle = False, num_workers= 4)

    eval_dataset = eval_Dataset(evalRGBD, eval_Labels)
    print('the number of evaluation dataset：', evalRGBD.__len__())
    #creat DataLoader
    evalloader = DataLoader.DataLoader(eval_dataset,batch_size= 1, shuffle = False, num_workers= 4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    mymodel = MyCNN(in_dim, n_class=894)
    # result=mymodel.forward(trainRGBD)
    mymodel = mymodel.to(device)
    mymodel.train_model(device)
    # mymodel.model_eval(device)

