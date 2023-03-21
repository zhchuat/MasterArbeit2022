import torch
import torch.nn as nn
import torchvision.models as models


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',nonlinearity='relu')
                # nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(_DecoderBlock, self).__init__()
        middle_channels = int(in_channels / 2)
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(middle_channels),
                      nn.ReLU(inplace=True),
                  ] * (num_conv_layers - 2)
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class SegNet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(SegNet, self).__init__()
        vgg = models.vgg19_bn()
        vgg19_bn_path = "vgg19_bn-c79401a0.pth"
        if pretrained:
            vgg.load_state_dict(torch.load(vgg19_bn_path))
        vgg = vgg.cuda()
        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:27])
        self.enc4 = nn.Sequential(*features[27:40])
        self.enc5 = nn.Sequential(*features[40:])

        self.dec5 = nn.Sequential(
            *([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)] +
              [nn.Conv2d(512, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True)] * 4)
        )
        self.dec4 = _DecoderBlock(1024, 256, 4)
        self.dec3 = _DecoderBlock(512, 128, 4)
        self.dec2 = _DecoderBlock(256, 64, 2)
        self.dec1 = _DecoderBlock(128, num_classes, 2)
        initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(torch.cat([enc4, dec5], 1))
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))
        dec1 = self.dec1(torch.cat([enc1, dec2], 1))
        return nn.Parameter(dec1)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import OrderedDict

# class SegNet(nn.Module):
#     def __init__(self,num_classes, input_nbr=4):
#         super(SegNet, self).__init__()

#         batchNorm_momentum = 0.1

#         self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
#         self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
#         self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

#         self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
#         self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

#         self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#         self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#         self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

#         self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

#         self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

#         self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

#         self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
#         self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#         self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

#         self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#         self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
#         self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
#         self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

#         self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
#         self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

#         self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
#         self.conv11d = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
#         initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1)


#     def forward(self, x):

#         # Stage 1
#         x11 = F.relu(self.bn11(self.conv11(x)))
#         x12 = F.relu(self.bn12(self.conv12(x11)))
#         x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)

#         # Stage 2
#         x21 = F.relu(self.bn21(self.conv21(x1p)))
#         x22 = F.relu(self.bn22(self.conv22(x21)))
#         x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

#         # Stage 3
#         x31 = F.relu(self.bn31(self.conv31(x2p)))
#         x32 = F.relu(self.bn32(self.conv32(x31)))
#         x33 = F.relu(self.bn33(self.conv33(x32)))
#         x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

#         # Stage 4
#         x41 = F.relu(self.bn41(self.conv41(x3p)))
#         x42 = F.relu(self.bn42(self.conv42(x41)))
#         x43 = F.relu(self.bn43(self.conv43(x42)))
#         x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)

#         # Stage 5
#         x51 = F.relu(self.bn51(self.conv51(x4p)))
#         x52 = F.relu(self.bn52(self.conv52(x51)))
#         x53 = F.relu(self.bn53(self.conv53(x52)))
#         x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)


#         # Stage 5d
#         x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
#         x53d = F.relu(self.bn53d(self.conv53d(x5d)))
#         x52d = F.relu(self.bn52d(self.conv52d(x53d)))
#         x51d = F.relu(self.bn51d(self.conv51d(x52d)))

#         # Stage 4d
#         x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
#         x43d = F.relu(self.bn43d(self.conv43d(x4d)))
#         x42d = F.relu(self.bn42d(self.conv42d(x43d)))
#         x41d = F.relu(self.bn41d(self.conv41d(x42d)))

#         # Stage 3d
#         x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
#         x33d = F.relu(self.bn33d(self.conv33d(x3d)))
#         x32d = F.relu(self.bn32d(self.conv32d(x33d)))
#         x31d = F.relu(self.bn31d(self.conv31d(x32d)))

#         # Stage 2d
#         x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
#         x22d = F.relu(self.bn22d(self.conv22d(x2d)))
#         x21d = F.relu(self.bn21d(self.conv21d(x22d)))

#         # Stage 1d
#         x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
#         x12d = F.relu(self.bn12d(self.conv12d(x1d)))
#         x11d = self.conv11d(x12d)

#         return x11d

#     def load_from_segnet(self, model_path):
#         s_dict = self.state_dict()# create a copy of the state dict
#         th = torch.load(model_path).state_dict() # load the weigths
#         # for name in th:
#             # s_dict[corresp_name[name]] = th[name]
#         self.load_state_dict(th)


if __name__ == '__main__':
    model = SegNet(38)
    print(model)
    model.to('cpu')
    data = torch.ones(1,4,640,480)
    data= data.to(device='cpu')
    output = model(data)
    print(output.shape)

    # seg_map = torch.argmax(output, dim=0)
    # seg_map = seg_map.detach().cpu().numpy().astype(np.uint8)
    # print(seg_map.shape)
