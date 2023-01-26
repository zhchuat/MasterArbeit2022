import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import torch

# img = np.array(Image.open("/home/hczhu/CNNlearn/dataset/NYU/RGBD/val/1163.png"))
img = Image.open("/home/hczhu/CNNlearn/dataset/NYU/RGBD/val/1163.png")
# 
# print(img.shape)
transf = transforms.ToTensor()
img_tensor = transf(img)
img_tensor = torch.reshape(img_tensor,(1,4,480,640))
print(img_tensor.shape)



from sim_CNN import *
from ClassNames import *
model = Sim_CNN(n_class=894)
model.load_state_dict(('/home/hczhu/CNNlearn/last.pt'),strict=False)

device = torch.device("cpu")
with torch.no_grad():
            
    outputs = model(img_tensor)
    # label image bearbeiten
    # eval_Labels = np.bincount(eval_Labels.flatten(),minlength= 894)

    # eval_Labels = torch.from_numpy(eval_Labels).float().to(device)/640/480 #output [batchsize,640,480] --> [batch size, 894]

    # _, predicted = torch.max(outputs.data, dim=0)
    _, predicted = torch.max(outputs, dim=0)
    class_num = predicted.numpy()
    
print(class_num) # 查看预测结果ID
print(CLASS_NAMES_894[class_num])                    
        #         total_predict += eval_Labels.size(0)
        #         correct_predict += (predicted == eval_Labels).sum().item()
                
        # print('Accuracy of the network on the test images: {}'.format(
        #         100.0 * correct_predict / total_predict))