#coding=utf-8
 
# # Create a config并配置要流​​式传输的管道
# # 颜色和深度流的不同分辨率
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
 
# # 开始流式传输
# profile = pipeline.start(config)
# # 获取深度传感器的深度标尺（参见rs - align示例进行说明）
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: ", depth_scale)
 
# # 创建对齐对象
# # rs.align允许我们执行深度帧与其他帧的对齐
# # “align_to”是我们计划对齐深度帧的流类型。
# align_to = rs.stream.color
# align = rs.align(align_to)
# index = 0
# color_path = '/dataset/rgb'
# depth_path = '/dataset/d'
 
# # Streaming循环
# try:
#     while True:
#         index += 1
#         # 获取颜色和深度的框架集
#         frames = pipeline.wait_for_frames()
#         # frames.get_depth_frame（）是640x360深度图像
 
#         # 将深度框与颜色框对齐
#         aligned_frames = align.process(frames)
 
#         # 获取对齐的帧
#         aligned_depth_frame = aligned_frames.get_depth_frame()# aligned_depth_frame是640x480深度图像
#         color_frame = aligned_frames.get_color_frame()
 
#         # 验证两个帧是否有效
#         if not aligned_depth_frame or not color_frame:
#             continue
 
#         depth_image = np.asanyarray(aligned_depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())
 
#         # 转换深度图像为三通道
#         depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
 
#         # 渲染图像
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_3d, alpha=0.03), cv2.COLORMAP_JET)
 
#         # 原图
#         cv2.namedWindow('color_image', cv2.WINDOW_AUTOSIZE)
#         cv2.imshow('color_image', color_image)
#         # 深度图
#         cv2.namedWindow('depth_colormap', cv2.WINDOW_AUTOSIZE)
#         cv2.imshow('depth_colormap', depth_colormap)
 
#         if index % 25 == 0:
#             color = 'depth_' + str(index) + '.jpg'
#             depth = 'depth_' + str(index) + '.jpg'
#             cv2.imwrite(os.path.join(color_path, color), color_image)
#             cv2.imwrite(os.path.join(depth_path, depth), depth_colormap)
 
#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('q') or key == 27:
#             cv2.destroyAllWindows()
#             break
# finally:
#     pipeline.stop()

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

pipeline = rs.pipeline()

#Create a config并配置要流​​式传输的管道
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

# 按照日期创建文件夹
save_path = os.path.join(os.getcwd(), "out", time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
os.mkdir(save_path)
os.mkdir(os.path.join(save_path, "color"))
os.mkdir(os.path.join(save_path, "depth"))

# 保存的图片和实时的图片界面
cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("save", cv2.WINDOW_AUTOSIZE)
saved_color_image = None # 保存的临时图片
saved_depth_mapped_image = None
saved_count = 0

# 主循环
try:
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_data = np.asanyarray(aligned_depth_frame.get_data(), dtype="float32")
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("live", np.hstack((color_image, depth_mapped_image)))
        key = cv2.waitKey(30)

        # s 保存图片
        if key & 0xFF == ord('s'):
            saved_color_image = color_image
            saved_depth_mapped_image = depth_mapped_image

            # 彩色图片保存为png格式
            cv2.imwrite(os.path.join((save_path), "color", "{}.png".format(saved_count)), saved_color_image)
            # 深度信息由采集到的float16直接保存为npy格式
            cv2.imwrite(os.path.join((save_path), "depth", "{}.png".format(saved_count)), depth_image)
            # np.save(os.path.join((save_path), "depth", "{}".format(saved_count)), depth_data)
            saved_count+=1
            cv2.imshow("save", np.hstack((saved_color_image, saved_depth_mapped_image)))

        # q 退出
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break    
finally:
    pipeline.stop()