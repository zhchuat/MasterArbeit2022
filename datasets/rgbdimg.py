import numpy as np
import os, os.path
import cv2
from PIL import Image
 
# 通过txt文件中的路径读取RGB图片
train_file = open('/home/hczhu/CNNlearn/dataset/NYU13/test_rgb.txt')
input_data_train = train_file.readlines()
train_file.close()
# 通过txt文件中的路径读取depth图
disp_file = open('/home/hczhu/CNNlearn/dataset/NYU13/test_depth.txt')
input_data_disp = disp_file.readlines()
disp_file.close()
 
for i, name in enumerate(input_data_train):
    # 去掉末尾的换行符，name = input_data_train[i]
    input_data_train[i] = input_data_train[i].strip()
    input_data_disp[i] = input_data_disp[i].strip()
    print (i)
    rgb = Image.open(input_data_train[i])
    disp = Image.open(input_data_disp[i])
    # disp = cv2.imread(input_data_disp[i],cv2. IMREAD_ANYCOLOR)
    rgb = np.array(rgb)
    disp = np.array(disp)
    rgbd = np.zeros((480, 640, 4), dtype=np.uint8)
    rgbd[:, :, 0] = rgb[:, :, 0]
    rgbd[:, :, 1] = rgb[:, :, 1]
    rgbd[:, :, 2] = rgb[:, :, 2]
    rgbd[:, :, 3] = disp
#     input_data_train[i] = os.path.split(input_data_train[i])
    # filename = '/home/hczhu/CNNlearn/dataset/NYU13/trainRGBD/%s' % i +'.png'
    file_name, extension = os.path.splitext(input_data_train[i])
    last_four = file_name[-4:]
    rgba_file = os.path.join('/home/hczhu/CNNlearn/dataset/NYU13/testRGBD/', last_four + '.png')
    # filename = '/home/hczhu/CNNlearn/dataset/NYU13/trainRGBD/%s' % i +'.png'

    print (rgba_file)

    imageRGBD = Image.fromarray(rgbd, mode='RGBA')
#     # 保存的RGBD图片文件名为RGB图片的文件名
    # cv2.imwrite(rgba_file, rgbd)
    imageRGBD.save(rgba_file)
print ("success")