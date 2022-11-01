
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms
from sim_CNN import *


#RGB to Tensor
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
#train and test dataset
train_set = torchvision.datasets.CIFAR10(root='./dataset/',
                                        train=True,
                                        transform=dataset_transform,
                                        download=False
                                        )
test_set = torchvision.datasets.CIFAR10(root='./dataset/',
                                        train=False,
                                        transform=dataset_transform,
                                        download=False)                                       

#data load
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

train_data_size = len(train_set)
test_data_size = len(test_set)
print("the train data size: {}".format(train_data_size))
print('the test data size: {}'.format(test_data_size))

CNN=sim_CNN()
#Loss function
loss_fn = nn.CrossEntropyLoss()
#optimizer
opt = torch.optim.Adam(CNN.parameters(),lr=0.001)

total_train_step = 0
total_test_step = 0

epoch = 5 

for i in range(epoch):
    print("----------the {} train beginning-------------".format(i+1))

    # training 
    for data in train_loader:
        imgs, classnames = data
        output = CNN(imgs)
        loss = loss_fn(output,classnames)

        opt.zero_grad()
        loss.backward()
        opt.step()
        total_train_step +=1
        print('total train step is: {}, Loss: {}'.format(total_train_step,loss.item()))


    #testing
    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, classnames = data
            output = CNN(imgs)
            loss = loss_fn(output,classnames)
            total_test_loss = total_test_loss + loss.item()
    print('tatal test loss:{}'.format(total_test_loss,))