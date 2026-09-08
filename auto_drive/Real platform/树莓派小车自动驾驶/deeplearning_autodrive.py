#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :deeplearning_autodrive.py
@说明        :基于深度学习的树莓派自动驾驶小车控制
@时间        :2022/03/25 20:15:30
@作者        :钱彬
@版本        :1.0
'''


# 导入库函数
import cv2
import numpy as np

# 导入openvino库
from openvino.inference_engine import IECore

# 导入小车运动库
from picarx import Picarx

# 导入自定义函数库
from tools import detect_lane, display_lines, compute_steer, stabilize_steering_angle, take_photo,ImageTrans

# 定义全局小车变量
px = Picarx()

# 创建并设置视频捕获对象
cap = cv2.VideoCapture(0)
print("摄像头是否已经打开 ？ {}".format(cap.isOpened()))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 设置图像宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # 设置图像高度

# 创建图像传输管道
img_trans = ImageTrans('192.168.2.148')


def main():
    '''
    主函数
    '''
    # 定义小车初始油门、转向角和状态
    speed = 0.1
    last_steer_angle = 0
    pic_index = 0

    # 设置模型路径
    model = "./ir/autodrive.xml"
    
    # 设置推理环境（CPU、GPU、MYRIAD）
    device = 'MYRIAD'  # 'MYRIAD'

    # 设置推理引擎
    ie = IECore() 
    
    # 读取网络模型
    net = ie.read_network(model=model)
    
    # 获取模型输入形状
    for input_key in net.input_info:
        if len(net.input_info[input_key].input_data.layout) == 4:
            n, c, h, w = net.input_info[input_key].input_data.shape

    # 定义运行精度
    for input_key in net.input_info:
        input_name = input_key
        net.input_info[input_key].precision = 'FP32'
        break

    # 准备执行网络
    exec_net = ie.load_network(network=net, device_name=device)

    # 循环控制
    while True:
        # 读取图像
        ret, frame = cap.read()
        if not ret:
            print("图像获取失败，请检查")
            break

        # 输入信息预处理
        images = np.ndarray(shape=(1, c, h, w))
        images_hw = []
        
        ih, iw = frame.shape[:-1]
        images_hw.append((ih, iw))
        img = cv2.resize(frame, (w, h))
        # BGR 转换为 HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # HWC 转换为 CHW 
        img = img.transpose((2, 0, 1))  
        # 转换至 (0,1)
        img = img / 255.0 
        images[0] = img 
        out_blob = next(iter(net.outputs))
        
        data = {}
        data[input_name] = images
 
        # 推理
        res = exec_net.infer(inputs=data)
        
        # 输出结果
        for i, probs in enumerate(res[out_blob]):
            probs = np.squeeze(probs)

        # 计算转向角
        steer_angle = probs

        # 平滑处理
        last_steer_angle = stabilize_steering_angle(last_steer_angle, steer_angle, 2, max_angle_deviation_two_lines=5)

        # 输出转向值
        print("\r 当前转向角度steer: %s    " % (last_steer_angle))

        # 执行动作
        px.set_dir_servo_angle(last_steer_angle)
        px.forward(speed)

        # 保存图像
        #if abs(last_steer_angle) > 12:
            #take_photo(last_steer_angle, frame, pic_index)
        pic_index += 1


if __name__ == "__main__":
    '''
    程序入口
    '''
    try:
        main()
    except Exception as e:
        print("error:%s" % e)
    finally:
        px.set_dir_servo_angle(0)
        px.forward(0)
        cap.release()
        cv2.destroyAllWindows()