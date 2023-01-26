import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

n_class = 894

class Sim_CNN(nn.Module):

    def __init__(self,n_class):
        super(Sim_CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4,64,3,1,1),      # 1st conv，in_channels:4，out_channels:32，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
            nn.MaxPool2d(2),            # 1st Max Pooling
            nn.BatchNorm2d(64),
            nn.SiLU(),
            # nn.Dropout(p=0.5, inplace=False),         
            
            nn.Conv2d(64,128,3,1,1),     # 2nd conv，in_channels:32，out_channels:64，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
            nn.MaxPool2d(2),            # 2nd Max Pooling
            nn.BatchNorm2d(128),
            nn.SiLU(),
            # nn.Dropout(p=0.5, inplace=False),
            
            nn.Conv2d(128,256,3,1,1),    # 3rd conv，in_channels:64，out_channels:128，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
            nn.MaxPool2d(2),            # 3rd Max Pooling
            nn.BatchNorm2d(256),
            nn.SiLU(),
            # nn.Dropout(p=0.5, inplace=False),
            
            nn.Conv2d(256,512,3,1,1),    # 3rd conv，in_channels:64，out_channels:128，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
            nn.MaxPool2d(2),            # 3rd Max Pooling
            nn.BatchNorm2d(512),
            nn.SiLU(),

            nn.Flatten(),
            nn.Linear(40*30*512,64),    # 1st fc with linear，in_features:80×80×128，out_features:64
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(64, n_class),          # 2nd fc with linear，in_features:64，out_features:10
            nn.Softmax(dim=1)
        )

    def forward(self,x):                # forward function to get the output of CNN, reference
        out = self.model(x)
        return out

# cnn = Sim_CNN(n_class)
# # #model testing
# print(cnn)
# input = torch.ones((16,4,480,640))
# print(input.shape)
# output = cnn(input)
# print(output.shape)