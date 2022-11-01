#sim CNN Modell

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

#device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu') #using CUDA
#torch.cuda.manual_seed(1)   # torch + GPU, 
                            # If not set, the initialization of each training is random, resulting in uncertain results


#RGB Modell
class sim_CNN(nn.Module):
    def __init__(self):
        super(sim_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,32,3,padding=1, dilation=1)  # 1st conv，in_channels:3，out_channels:32，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
        self.maxpool1 = nn.MaxPool2d(2)                             # 1st Max Pooling
        self.conv2 = torch.nn.Conv2d(32,32,3,padding=1, dilation=1) # 2nd conv，in_channels:32，out_channels:32，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
        self.maxpool2 = nn.MaxPool2d(2)                             # 2nd Max Pooling
        self.conv3 = torch.nn.Conv2d(32,64,3,padding=1, dilation=1) # 3rd conv，in_channels:32，out_channels:64，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
        self.maxpool3 = nn.MaxPool2d(2)                             # 3rd Max Pooling
        self.flatten=nn.Flatten()
        self.fc1 = nn.Linear(4*4*64,64)                             # 1st fc with linear，in_features:4×4×64，out_features:128
        self.fc2 = nn.Linear(64, 10)                                # 2nd fc with linear，in_features:64，out_features:10

    def forward(self, x):           # forward function to get the output of CNN, reference
        #conv1
        x = self.conv1(x)           # the 1st conv
        #x = F.relu(x)               # the result of 1st conv is dealt with ReLU activate function
        x = self.maxpool1(x)        # 1st pooling，pooling size 4×4，Max pooling
        #conv2
        x = self.conv2(x)           # the 2nd conv
        #x = F.relu(x)               # the result of 2nd conv is dealt with ReLU activate function
        x = self.maxpool2(x)        # 2nd pooling，pooling size 4×4，Max pooling
        #conv3
        x = self.conv3(x)           # the 3rd conv
        #x = F.relu(x)               # the result of 3rd conv is dealt with ReLU activate function
        x = self.maxpool3(x)        # 3rd pooling，pooling size 4×4，Max pooling
        #fc
        #x = x.view(x.size()[0],-1)  # the fc output with 1-dim，so the input format is [4×4×64] ---> [1024×1]
        x = self.flatten(x)
        #x = F.relu(self.fc1(x))     # 1sf FC，ReLU activate
        x = self.fc1(x)
        #x = F.relu(self.fc2(x))     # 2nd FC，ReLU activate
        x = self.fc2(x)

        return x  

if __name__ == '__main__':
    CNN=sim_CNN()
    #model testing
    input = torch.ones((64,3,32,32))
    output = CNN(input)
    print(output.shape)
    print(CNN)



#model output using Tensorboard, open with tensorboard --logdir=log_result
# writer = SummaryWriter('./log_result/')
# writer.add_graph(CNN, input)
# writer.close()

#model output using ONNX
# model_out_path = "./log_result/CNNmodel.pt"
# torch.save(CNN, model_out_path)
# print("the Model has been saved to {}".format(model_out_path))
