
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from Model.CGNet import Context_Guided_Network
from Model.FCNNet import FCN8
from Model.LEDNet import LEDnet
# from SUN_dataset3c import TrainDataset
# from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(42)  # reproducible

# from ClassNames_SUN import *
from ClassNames_NYU import *

# CLASS_COLORS_894 = get_colormap(1+894).tolist()
# data_path = "/home/zhu/dataset/SUN/"
num_classes=14+1
""" Eval function --------------------------------------------------------------------""" 
# def model_eval(self,device):

with torch.no_grad():
    # 选择设备，有cuda用cuda，没有就用cpu
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
    # 加载网络，图片单通道，分类为1。
    # model = FCN8(num_classes=num_classes)
    # model = LEDnet(num_classes=num_classes)
    model = Context_Guided_Network(classes=num_classes,M=3,N=21)
    # 将网络拷贝到deivce中
    # 加载模型参数
    checkpoint = torch.load('/home/hczhu/CNNlearn/ckpt_epoch_57.pth',
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)    
    # 测试模式
    model.eval()
    cmap = np.array(ColorsMap).astype('uint8')

    img_name = '0462.png'
    # img_name = '0001.png'
    
    rgb='/home/hczhu/CNNlearn/dataset/NYU13/test_rgb/'+ img_name
    depth='/home/hczhu/CNNlearn/dataset/NYU13/test_depth/'+ img_name
    Img_path = '/home/hczhu/CNNlearn/dataset/NYU13/testRGBD/'+ img_name
    Label_path = '/home/hczhu/CNNlearn/dataset/NYU13/test_label13/'+ img_name

    # rgb='/home/zhu/dataset/NYU13/IMG/test_rgb/'+ img_name
    # depth='/home/zhu/dataset/NYU13/IMG/test_depth/'+ img_name
    # Img_path = '/home/zhu/dataset/NYU13/testRGBD/'+ img_name
    # Label_path = '/home/zhu/dataset/NYU13/test_label13/'+ img_name

    img = Image.open(Img_path)

    transform_train = transforms.Compose([  
                                        # transforms.Resize((480,640)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=INPUT.PIXEL_MEAN)
                                        # transforms.Normalize((0.472, 0.427, 0.461, 0.538), (0.298, 0.279, 0.290, 0.270))
                                        # transforms.Normalize((0.475, 0.404, 0.385, 0.309), (0.290, 0.296, 0.310, 0.187)) 
                                        transforms.Normalize((0.475, 0.404, 0.385, 0.398), (0.290, 0.297, 0.310, 0.178)) #eval
                                        # transforms.Normalize((0.485, 0.416, 0.398, 0.409), (0.288, 0.295, 0.309, 0.187))#train
                                        ])
    img_tensor = transform_train(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)                                      
    out = model(img_tensor)
    print('out', out.shape)

    pred = torch.argmax(out, axis=1)
    # print('argmax', pred.shape)
    pred = pred.cpu().numpy().squeeze(0).astype(np.uint8)
    print('pred', pred.shape)
    index_pred=np.array(np.unique(pred))
    print('Pred',index_pred)
    pred_Object =[]
    for i in index_pred:
        pred_Object.append(CLASS_NAMES_14[i])
        # print(i,CLASS_NAMES_GERMAN[i])
    print(pred_Object)

    # seg_img = np.zeros((480, 640,3))
    colored_segmentation = cmap[pred].astype(np.uint8)

    label = np.array(Image.open(Label_path))
    print('label', label.shape)
    index_label=np.array(np.unique(label))
    print('label',index_label)

    colored_label = cmap[label].astype(np.uint8)

    #draw subplot RGB D RGBD GT Pred
    plt.figure(figsize=(25, 5), dpi=100)
    grid = plt.GridSpec(4,15,wspace=0.1,hspace=0.01)
    plt.subplot(grid[0:3,0:3]).set_title('RGB')
    rgb = np.array(Image.open(rgb))
    plt.axis("off")
    plt.imshow(rgb)

    plt.subplot(grid[0:3,3:6]).set_title('Depth')
    depth = np.array(Image.open(depth))
    plt.axis("off")
    plt.imshow(depth)

    plt.subplot(grid[0:3,6:9]).set_title('RGB-D')
    rgbd = np.array(Image.open(Img_path))
    plt.axis("off")
    plt.imshow(rgbd)


    plt.subplot(grid[0:3,9:12]).set_title('Ground Truth')
    plt.axis("off")
    plt.imshow(colored_label)

    plt.subplot(grid[0:3,12:15]).set_title('Pred')
    plt.axis("off")
    plt.imshow(colored_segmentation)

    #draw a Colors Map
    for i,color in enumerate(ColorsMap):
        plt.subplot(grid[3,i:i+1]).set_title(CLASS_NAMES_14[i])
        color=np.array([[color]])
        plt.axis("off")
        plt.imshow(color)
        
    plt.show()
    # cnt = 0
    # for k, v in model.state_dict().items():
    #     print(k, v.size(), torch.numel(v))
    #     cnt += torch.numel(v)
    # print('total parameters:', cnt)
    # pred = out.max(1).squeeze(0).cpu().data.numpy()
    
    # cv2.imwrite('/home/hczhu/ESANet/src/datasets/sunrgbd/result/'+'res.png',colored_segmentation) #保存可视化图像
    # cv2.imwrite(savedir+'val_pred_result/'+'res.png',x_visualize) #保存可视化图像
print('finish')