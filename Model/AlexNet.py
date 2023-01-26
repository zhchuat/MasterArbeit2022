import torch
import torch.nn as nn
import torchvision

class AlexNet(nn.Module):
    def __init__(self,num_classes=894):
        super(AlexNet,self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=4,out_channels=96,kernel_size=11,stride=4,padding=2,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
            nn.Conv2d(in_channels=96,out_channels=192,kernel_size=5,stride=1,padding=2,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
            nn.Conv2d(in_channels=192,out_channels=384,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*12*12,out_features=4096),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
    def forward(self,x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0),256*13*13)
        x = self.classifier(x)
        return x


if __name__ =='__main__':
    # model = torchvision.models.AlexNet()
    model = AlexNet()
    print(model)

    input = torch.randn(1,4,448,448)
    out = model(input)
    print(out.shape)