
from pydoc import classname
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

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

test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# #show the img in tensorboard, usage: tensorboard --logdir=dataloader
writer = SummaryWriter('dataloader')  # type: ignore
step = 0
for data in test_loader:
    imgs, classnames = data
    print(imgs.shape)
    print(classnames)
    writer.add_images('test_set_drop_last',imgs,step)
    step+=1
writer.close()
#
# img, classnames = test_set[0]
# print(img.shape)
# print(classnames)

# for data in test_loader:
#     imgs, classnames = data
#     print(imgs.shape)
#     print(classnames)
    




# for i in range(10):
#     img, classnames = test_set[i]
#     writer.add_image('test_set',img,i)


# img, target = test_set[0]
# print(img)
# print(test_set.classes[target])
# img.show()












# Imagesize_H = 640
# Imagesize_W = 480

# transforms = transforms.Compose([
#     transforms.Resize(256),    # 将图片短边缩放至256，长宽比保持不变：
#     transforms.CenterCrop(224),   #将图片从中心切剪成3*224*224大小的图片
#     transforms.ToTensor()          #把图片进行归一化，并把数据转换成Tensor类型
# ])

# class MyDataset(Dataset):
#     def __init__(self, img_path, transform=None):
#         super(MyDataset, self).__init__()
#         self.root = img_path
 
#         self.txt_root = self.root + '\\' + 'data.txt'
 
#         f = open(self.txt_root, 'r')
#         data = f.readlines()
 
#         imgs = []
#         labels = []
#         for line in data:
#             line = line.rstrip()
#             word = line.split()
#             #print(word[0], word[1], word[2])   
#             #word[0]是图片名字.jpg  word[1]是label  word[2]是文件夹名，如sunflower
#             imgs.append(os.path.join(self.root,word[2], word[0]))
 
#             labels.append(word[1])
#         self.img = imgs
#         self.label = labels
#         self.transform = transform
 
#     def __len__(self):
#         return len(self.label)
 
#     def __getitem__(self, item):
#         img = self.img[item]
#         label = self.label[item]
 
#         img = Image.open(img).convert('RGB')
 
#         # 此时img是PIL.Image类型   label是str类型
 
#         if self.transform is not None:
#             img = self.transform(img)
 
#         label = np.array(label).astype(np.int64)
#         label = torch.from_numpy(label)
 
#         return img, label

# path = r'./COCO/trainRGB/'
# dataset = MyDataset(path, transform=transforms)
 
# data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
 
 
# for i, data in enumerate(data_loader):
#     images, labels = data
 
#     # 打印数据集中的图片
#     img = torchvision.utils.make_grid(images).numpy()
#     plt.imshow(np.transpose(img, (1, 2, 0)))
#     plt.show()
 
#     break
