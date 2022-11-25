import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter

transform_train = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     #transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_val = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class MyDataSet(Dataset):
    """自定义数据集"""
    # def __init__(self, root,transform):
    def __init__(self, root='../dataset/', transform=True):
        self.root = root
        
        self.images_path = torch.tensor(np.load(os.path.join(root,"RGBD.npy")))
        self.images_class = torch.tensor(np.load(os.path.join(root,"labels.npy")))
        self.transform = transform
 
    def __len__(self):
        return self.images_path.shape[0]  # 返回数据的总个数
  
    def __getitem__(self, index):
        img = self.images_path[index, :, :]  # 读取每一个npy的数据
        label = self.images_class[index]  # lesen jede npy
        img = np.expand_dims(img, axis=0)
        img = torch.Tensor(img)
        img = torch.cat([img, img, img], dim=0)
        ###############################################################################################################3
        ###############################################################################################################3
        label = label.type(torch.long)
 
        if self.transform is not None:
            img = self.transform(img)
        return img, label  # 返回数据还有标签

class MyDataset(Dataset):
    def __init__(self, root='../dataset/', transform=True):
        self.train_data = np.load(os.path.join(root,"RGBD.npy")) #加载RGBD.npy数据
        self.train_data = np.load(os.path.join(root,"labels.npy")) #加载RGBD.npy数据
        self.transforms = transform_train #转为tensor形式

    def __getitem__(self, index):
        images= self.train_data[index, :, :, :]  # 读取每一个npy的数据
        # hdct = np.squeeze(hdct)  # 删掉一维的数据，就是把通道数这个维度删除
        # ldct = 2.5 * skimage.util.random_noise(hdct * (0.4 / 255), mode='poisson', seed=None) * 255 #加poisson噪声
        images=Image.fromarray(np.uint8(images)) #转成image的形式
        labels=Image.fromarray(np.uint8(labels)) #转成image的形式
        images= self.transforms(images)  #转为tensor形式
        labels= self.transforms(labels)  #转为tensor形式
        return labels,images #返回数据还有标签

    def __len__(self):
        return self.train_data.shape[0] #返回数据的总个数
 
    def main():
        dataset=MyDataset('../dataset/')
        train_data= DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=0)
    
    if __name__ == '__main__':
        main()





trainloader = DataLoader(batch_size=16, shuffle=True, num_workers=0)

testloader = DataLoader(batch_size=16, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class MyCNN(nn.Module):

    def __init__(self):
        super(MyCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4,32,3,1,1),      # 1st conv，in_channels:4，out_channels:32，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
            nn.MaxPool2d(2),            # 1st Max Pooling
            nn.BatchNorm2d(32),
            nn.SiLU(),         
            
            nn.Conv2d(32,64,3,1,1),     # 2nd conv，in_channels:32，out_channels:64，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
            nn.MaxPool2d(2),            # 2nd Max Pooling
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            nn.Conv2d(64,128,3,1,1),    # 3rd conv，in_channels:64，out_channels:128，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
            nn.MaxPool2d(2),            # 3rd Max Pooling
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            nn.Flatten(),
            nn.Linear(80*80*128,64),    # 1st fc with linear，in_features:80×80×128，out_features:64
            nn.Linear(64, 10),          # 2nd fc with linear，in_features:64，out_features:10
        )

    def forward(self,x):                # forward function to get the output of CNN, reference
        self.model(x)
        return self.model(x)


    def train_model(self,device):
        #creat optimizer
        optimizer = optim.Adam(self.parameters(), lr=0.001)
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
            
            for data in MyDataset(trainloader):
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
mymodel = MyCNN()
mymodel = mymodel.to(device)
mymodel.train_model(device)
mymodel.model_eval(device)
torch.save(mymodel.state_dict(), 'mymodel.pt') #save model