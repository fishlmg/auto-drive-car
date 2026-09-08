#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :auto_drive.py
@说明        :自动驾驶小车控制(基于端到端深度学习模型)
@版本        :1.0
'''


# 导入系统库
import cv2
import numpy as np
import math
import gym
import gym_donkeycar

# 导入PyTorch库
from torch import nn
import torch

# 导入自定义库
from models import AutoDriveNet
from utils import *


def main():
    '''
    主函数
    '''
    # 设置模拟器环境
    env = gym.make("donkey-generated-roads-v0")
    
    # 设置推理环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型
    checkpoint = torch.load('D:/program/auto_drive/Simulation platform/deep_learning/results/checkpoint.pth', map_location=torch.device('cpu'))

    model = AutoDriveNet()
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    
    # # 设置归一化参数
    # PIXEL_MEANS = (0.485, 0.456, 0.406)  # RGB格式的均值和方差
    # PIXEL_STDS = (0.229, 0.224, 0.225)

    # 重置当前场景
    obv = env.reset()

    # 开始启动
    action = np.array([0, 0.1])  # 动作控制，第1个转向值，第2个油门值

    # 执行动作并获取图像
    img, reward, done, info = env.step(action)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # 运行5000次动作
    model.eval()
    for t in range(5000):
        
        # 图像预处理
        img = torch.from_numpy(img.copy()).float()
        img /= 255.0
        # img -= torch.tensor(PIXEL_MEANS)
        # img /= torch.tensor(PIXEL_STDS)
        img = img.permute(2, 0, 1)
        img.unsqueeze_(0)

        # 转移数据至设备
        img = img.to(device)

        # 模型推理
        steering_angle = 0
        factor=1.0
        with torch.no_grad():
            # 计算转向角度
            steering_angle = (model(img).squeeze(0).cpu().detach().numpy())[0]
            if steering_angle*factor<-1:
                steering_angle=-1
            elif steering_angle*factor>1:
                steering_angle=1
            else:
                steering_angle=steering_angle*factor
            print(steering_angle)
            action = np.array([steering_angle, 0.1])  # 油门值恒定
            
            # # 手动释放内存
            # del img

            # 执行动作并更新图像
            img, reward, done, info = env.step(action)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


    # 运行完以后重置当前场景
    obv = env.reset()


if __name__ == '__main__':
    '''
    主函数入口
    '''
    main()
