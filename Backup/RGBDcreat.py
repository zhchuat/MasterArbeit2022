import numpy as np
import os, os.path
import cv2
import torch
 
# from txt path read RGB.npy
train_rgb_file = open('/home/hczhu/CNNlearn/dataset/NYU-Depth_V2/training/images.txt','r')
input_data_rgb = train_rgb_file.readlines()
train_rgb_file.close()
# from txt path read depth.npy
train_depth_file = open('/home/hczhu/CNNlearn/dataset/NYU-Depth_V2/training/depths.txt','r')
input_data_depth = train_depth_file.readlines()
train_depth_file.close()
# from txt path read labels.npy
train_labels_file = open('/home/hczhu/CNNlearn/dataset/NYU-Depth_V2/training/labels.txt','r')
input_data_labels = train_labels_file.readlines()
train_labels_file.close()

rgbd_dataset = []
labels_dataset = []
for i, name in enumerate(input_data_rgb):
    # name = input_data_train[i]
    input_data_rgb[i] = input_data_rgb[i].strip()
    input_data_depth[i] = input_data_depth[i].strip()
    input_data_labels[i] = input_data_labels[i].strip()
    
    # rgb = cv2.imread(input_data_train[i],cv2. IMREAD_ANYCOLOR)
    # disp = cv2.imread(input_data_disp[i],cv2. IMREAD_ANYCOLOR)
    # rgb = np.array(rgb)
    # disp = np.array(disp)

    loadrgb=np.load(input_data_rgb[i])
    loaddepth=np.load(input_data_depth[i])
    loadlabels=np.load(input_data_labels[i])
    # print(loadrgb.shape)
    # print(loaddepth.shape)

    rgbd = np.zeros((4, 640, 480), dtype=np.uint8)
    rgbd[0, :, :] = loadrgb[0, :, :]
    rgbd[1, :, :] = loadrgb[1, :, :]
    rgbd[2, :, :] = loadrgb[2, :, :]
    rgbd[3, :, :] = loaddepth
      
    labels = np.zeros((640, 480), dtype=np.uint8)
    labels[:] = loadlabels[:]
    
    rgbd_dataset.append(rgbd)
    labels_dataset.append(labels)
    # print(rgbd)
    i+=1
    
    # input_data_train[i] = os.path.split(input_data_train[i])
    #filename = '../training/RGBD/%s' % i +'.npy'
    #filename = './RGBD.npy'

    #print (filename)
    # Save RGBD Image filename as RGB Image Filename
    
    

RGBD = np.array(rgbd_dataset)
LABEL =  np.array(labels_dataset)   
print(RGBD.shape)
print(RGBD[0, :].shape)
print ("success")
np.save('../RGBD.npy', RGBD)
np.save('../labels.npy', LABEL)