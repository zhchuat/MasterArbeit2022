import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Sim_CNN(nn.Module):

    def __init__(self):
        super(Sim_CNN, self).__init__()
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

cnn = Sim_CNN()
# #model testing
input = torch.ones((8,4,640,640))
print(input.shape)
output = cnn(input)
print(output.shape)
print(cnn)