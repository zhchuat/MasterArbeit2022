#import bibpackage

import torch
import os,glob
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader

torch.manual_seed(42)  # reproducible


'''creat train dataset'''

class TrainDataset(Dataset.Dataset):
    def __init__(self, data_path):
        # Initial，read all images and labels from data_path
        self.data_path = data_path
        self.RGBDs_path = glob.glob(os.path.join(data_path, 'SUN_rgbd/*.png'))
        self.labels_path = glob.glob(os.path.join(data_path, 'SUN_label/*.png'))

  
    def __getitem__(self, index):
        # from index import RGBDImage and Labels 
        image_path = self.RGBDs_path[index]
        label_path = self.labels_path[index]
        # show training RGBD and labels Images
        image = Image.open(image_path)
        label = Image.open(label_path)
        transform_train = transforms.Compose([  
                                            transforms.Resize((480,640)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(15),
                                            transforms.ToTensor(),
                                            ])
        transform_train_Norm = transforms.Compose([
                                            # transforms.Normalize((0.480, 0.410, 0.392, 0.278), (0.289, 0.295, 0.308, 0.139))，
                                            transforms.Normalize((0.517, 0.508, 0.475, 0.3), (0.266, 0.281, 0.278,0.2)) 
                                            ])                                                                       
        image_tensor=  transform_train(image)
        image_tensor=  transform_train_Norm(image_tensor)
        label_tensor=  transform_train(label)

        return image_tensor, label_tensor

    def __len__(self):
        # Traning dataset size
        return len(self.RGBDs_path)


'''creat evaluation dataset'''

# class EvalDataset(Dataset.Dataset):
#     def __init__(self, data_path):
#         # Initial，read all images and labels from data_path
#         self.data_path = data_path
#         self.RGBDs_path = glob.glob(os.path.join(data_path, 'RGBD/*.png'))
#         self.labels_path = glob.glob(os.path.join(data_path, 'GT/*.png'))
 
#     def __getitem__(self, index):
#         # from index import RGBDImage and Labels 
#         image_path = self.RGBDs_path[index]
#         label_path = self.labels_path[index]
#         # show Eval RGBD and labels Images
#         image = Image.open(image_path)
#         label = Image.open(label_path)
#         transform_Eval = transforms.Compose([
#                                             transforms.ToTensor(),
#                                             ])
#         transform_Eval_Norm = transforms.Compose([
#                                             transforms.Normalize((0.480, 0.410, 0.392, 0.278), (0.289, 0.295, 0.308, 0.139)) ])                                                                       
#         image=  transform_Eval(image)
#         image=  transform_Eval_Norm(image)
#         label=  transform_Eval(label)                 

#         return image, label

#     def __len__(self):
#         # Traning dataset size
#         return len(self.RGBDs_path)

if __name__ == "__main__":
    batchsize = 2
    train_dataset = TrainDataset("/home/hczhu/ESANet/src/datasets/sunrgbd/")
    print("total train dataset size：", len(train_dataset))
    train_loader = DataLoader.DataLoader(dataset=train_dataset,
                                               batch_size=batchsize, 
                                               shuffle=True,
                                               num_workers= 0)

    # eval_dataset = EvalDataset("/home/hczhu/ESANet/src/datasets/sunrgbd")
    # print("total eval dataset size：", len(eval_dataset))
    # eval_loader = DataLoader.DataLoader(dataset=eval_dataset,
    #                                            batch_size=batchsize, 
    #                                            shuffle=True,
    #                                            num_workers= 0)
    
    for i, item in enumerate(train_loader):
        print(i)
        image, label = item
        print(image.shape)
        print(label.shape)
