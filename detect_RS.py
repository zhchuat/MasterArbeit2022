import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

from ClassNames_NYU import *
from Model.CGNet import Context_Guided_Network
num_classes=14+1

model = Context_Guided_Network(classes=num_classes,M=3,N=21)
checkpoint = torch.load('/home/hczhu/CNNlearn/ckpt_epoch_39.pth',
                            map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'], strict=True)

# Configure depth and color streams
device = torch.device("cpu")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data(), dtype="float32")
        max =depth_image.max()
        depth_image = (depth_image / max-0.5)/0.5
        
        # Stack both images horizontally
        img_tensor = (torch.from_numpy(color_image)/255-0.5)/0.5
        # img_tensor = transforms.ToTensor(img_tensor)
        depth_tensor = torch.from_numpy(depth_image)
        
        depth_tensor = torch.unsqueeze(depth_tensor, dim=2)
        input = torch.cat((img_tensor,depth_tensor), dim=2)
        input = input.view(1,4,480,640).to(device)
                     
        results = model(input)
        # print('out', results.shape)

        pred = torch.argmax(results, axis=1)
        # print('argmax', pred.shape)
        pred = pred.cpu().numpy().squeeze(0).astype(np.uint8)
        # print('pred', pred.shape)
        index_pred=np.array(np.unique(pred))
        print('Pred',index_pred)
        pred_Object =[]
        for i in index_pred:
            pred_Object.append(CLASS_NAMES_14[i])
            # print(i,CLASS_NAMES_GERMAN[i])
        print(pred_Object)
        # boxs= results.pandas().xyxy[0].values
        #boxs = np.load('temp.npy',allow_pickle=True)
        # dectshow(color_image,  depth_image)#boxs,
        cmap = np.array(ColorsMap).astype('uint8')
        colored_segmentation = cmap[pred].astype(np.uint8)
        #Show images     
        
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RealSense", np.hstack((color_image, colored_segmentation)))
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    # Stop streaming
    pipeline.stop()