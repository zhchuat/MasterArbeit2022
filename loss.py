import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import numpy as np
# 构造样本 假设一共有四种类别，图像大小是5x5的，batch_size是1
inputs_tensor = torch.randn(1, 4, 640, 480)

# inputs_tensor = torch.unsqueeze(inputs_tensor,0)
# inputs_tensor = torch.unsqueeze(inputs_tensor,1)
print('--------/n input size(nBatch x nClasses x height x width): ', inputs_tensor.shape)

targets_tensor = torch.randn(1, 640, 480) 
# targets_tensor = torch.LongTensor(np.array([[1, 2, 3], [4, 5, 6]])) 
# targets_tensor = torch.unsqueeze(targets_tensor,0)
print ('--------/n target size(nBatch x height x width): ', format(targets_tensor.shape))
 



model = nn.Sequential(
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
    nn.Linear(80*60*128,64),    # 1st fc with linear，in_features:80×80×128，out_features:64
    nn.Linear(64, 894),          # 2nd fc with linear，in_features:64，out_features:10
    # nn.Linear(640,480),
    
)

model_label = nn.Sequential(
    nn.Conv2d(1,32,3,1,1),      # 1st conv，in_channels:4，out_channels:32，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
    nn.MaxPool2d(2),            # 1st Max Pooling
    nn.BatchNorm2d(32),
    nn.SiLU(),         
    
    nn.Conv2d(32,64,3,1,1),     # 2nd conv，in_channels:32，out_channels:64，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
    nn.MaxPool2d(2),            # 2nd Max Pooling
    nn.BatchNorm2d(64),
    nn.SiLU(),
    
    nn.Conv2d(64,128,3,1,1),    # 3rd conv，in_channels:64，out_channels:128，conv_size:3×3，padding:1，dilation:1 (without Dilated Conv)
    nn.MaxPool2d(2),           # 3rd Max Pooling
    nn.BatchNorm2d(128),
    nn.SiLU(),
    
    nn.Flatten(),
    nn.Linear(80*60*128,64),    # 1st fc with linear，in_features:80×80×128，out_features:64
    nn.Linear(64, 894),          # 2nd fc with linear，in_features:64，out_features:10
    # nn.Linear(640,480),
    
)

input_img = model(inputs_tensor)
print(input_img.shape)


# targets_variable = autograd.Variable(targets_tensor)
targets_tensor = torch.unsqueeze(targets_tensor,0)
print(targets_tensor.shape)

label=model_label(targets_tensor)
print(label.shape)

loss = nn.CrossEntropyLoss()
output = loss(input_img, label)

print ('--NLLLoss2d: {}'.format(output))