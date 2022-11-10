import numpy as np
import os, os.path
import cv2
 
# 通过txt文件中的路径读取RGB图片
train_file = open('./train_rgb.txt')
input_data_train = train_file.readlines()
train_file.close()
# 通过txt文件中的路径读取视差图
disp_file = open('./train_depth.txt')
input_data_disp = disp_file.readlines()
disp_file.close()
 
for i, name in enumerate(input_data_train):
    # 去掉末尾的换行符，name = input_data_train[i]
    input_data_train[i] = input_data_train[i].strip()
    input_data_disp[i] = input_data_disp[i].strip()
    print (i)
    rgb = cv2.imread(input_data_train[i],cv2. IMREAD_ANYCOLOR)
    disp = cv2.imread(input_data_disp[i],cv2. IMREAD_ANYCOLOR)
    rgb = np.array(rgb)
    disp = np.array(disp)
    rgbd = np.zeros((530, 730, 4), dtype=np.uint8)
    rgbd[:, :, 0] = rgb[:, :, 0]
    rgbd[:, :, 1] = rgb[:, :, 1]
    rgbd[:, :, 2] = rgb[:, :, 2]
    rgbd[:, :, 3] = disp
#     input_data_train[i] = os.path.split(input_data_train[i])
    filename = './img_txt/%s' % i +'.png'
    print (filename)
#     # 保存的RGBD图片文件名为RGB图片的文件名
    cv2.imwrite(filename, rgbd)
print ("success")