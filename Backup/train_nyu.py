import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import torchvision.transforms as transforms
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter


transform_train = transforms.Compose([
                                        transforms.RandomCrop(480),
                                        transforms.RandomHorizontalFlip(),
                                        #transforms.RandomHorizontalFlip(),
                                        #transforms.RandomGrayscale(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
                                        ])

transform_val = transforms.Compose([
    
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])




class TrainDataset(Dataset):
    def __init__(self, data_path='../dataset/', transform=True): # NYU-Depth_V2/train/
        self.train_images = np.load(os.path.join(data_path,"RGBD.npy")) #加载RGBD.npy数据
        self.train_labels = np.load(os.path.join(data_path,"labels.npy")) #加载RGBD.npy数据
        # 查看图片与标签数量是否一致
        assert len(self.train_images) == len(self.train_labels), 'Number does not match'
        self.transforms = transform #转为tensor形式
                
        self.images_and_labels = []  # 创建一个空列表
        for i in range(len(self.train_images)):  # 往空列表里装东西，为了之后索引
            self.images_and_labels.append(
                (data_path + '/RGBD/' + self.train_images[i], data_path + '/labels/' + self.train_labels[i])
            )



    def __getitem__(self, index):
        # 读取数据
        image_path, label_path = self.images_and_labels[index]
        # 图像处理
        image = cv2.imread(image_path)  # 读取图像,(H,W,C)
        image = cv2.resize(image, (640, 640))  # 将图像尺寸变为 640*640
        # 对标签进行处理，从而可以与结果对比，得到损失值
        label = cv2.imread(label_path, 0)  # 读取标签，且为灰度图
        label = cv2.resize(label, (640, 640))  # 将标签尺寸变为 224*224


        images= self.train_data[index, :, :, :]  # 读取每一个npy的数据
        # hdct = np.squeeze(hdct)  # 删掉一维的数据，就是把通道数这个维度删除
        # ldct = 2.5 * skimage.util.random_noise(hdct * (0.4 / 255), mode='poisson', seed=None) * 255 #加poisson噪声
        images=Image.fromarray(np.uint8(images)) #转成image的形式
        labels=Image.fromarray(np.uint8(labels)) #转成image的形式
        images= self.transforms(images)  #转为tensor形式
        labels= self.transforms(labels)  #转为tensor形式
        return labels,images #返回数据还有标签

    def __len__(self):
        return self.train_images.shape[0] #返回数据的总个数
\



class TestDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.test_images = os.listdir(data_path + '/images')
        self.transform = transform
        self.imgs = []
        for i in range(len(self.images)):
            # self.imgs.append(data_path + '/images/' + self.images[i])
            self.imgs.append(os.path.join(data_path, 'images/', self.test_images[i]))
 
    def __getitem__(self, item):
        img_path = self.imgs[item]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
 
        if self.transform is not None:
            img = self.transform(img)
        return img
 
    def __len__(self):
        return len(self.test_images)

    
if __name__ == '__main__':
    img = cv2.imread('/home/hczhu/CNNlearn/dataset/NYU/nyu_images/10.jpg')
    img = cv2.resize(img, (640, 640))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img2 = img / 255
    cv2.imshow('pic1', img2)
    cv2.waitKey()
    print(img2)

    img3 = img2.astype('uint8')
    cv2.imshow('pic2', img3)
    cv2.waitKey()
    print(img3)

    # 下面开始矩阵就变成了3维
    hot1 = np.eye(2)[img3]  # 对标签矩阵的每个元素都做了编码，(0,1)背景元素，(1,0)目标元素
    print(hot1)
    print(hot1.ndim)
    print(hot1.shape)  # (16,16,2) C=16,H=16,W=16

    hot2 = np.array(list(map(lambda x: abs(x - 1), hot1))) # 变换一下位置。(1,0)背景元素，(0,1)目标元素
    print(hot2)
    print(hot2.ndim)
    print(hot2.shape)  # (16,16,2) C=16,H=16,W=16

    #hot3 = hot2.transpose(2, 0, 1)
    #print(hot3)  # (C=2,H=16,W=16)

'''
Dataloader
'''
#TrainDataset="./"
tensorform=torchvision.transforms.Compose([torchvision.transform.ToTesor()])
img=tensorform(img)
img=img.reshape(1,4,640,640)
import torchvision
trainloader = DataLoader(root=TrainDataset,batch_size=16, transform=torchvision.transforms.ToTensor(),shuffle=True, num_workers=0)

testloader = DataLoader(batch_size=16, shuffle=False, num_workers=0)




'''
CNN Modell 
input 4 chanal RGBD
'''

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
        #creat optimizerimages
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
            
            for data in TrainDataset(trainloader):
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
                running_loss =+ train_loss
                if total_train_step % 100 == 0:
                    print('total train step is: {}, Loss: {}'.format(total_train_step,running_loss))
                    writer.add_scalar("train_loss", running_loss, total_train_step)

            print("Epoch: {} Cost: {} Time: sec {}".format(epoch+1, running_loss, time.time()-timestart))
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
x=torch.range(0,64*4).view(1,4,8,8)
result=mymodel.forward(img)

mymodel = mymodel.to(device)
mymodel.train_model(device)
mymodel.model_eval(device)
torch.save(mymodel.state_dict(), 'mymodel.pt') #save model