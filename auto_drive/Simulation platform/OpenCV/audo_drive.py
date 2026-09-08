#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :audo_drive.py
@说明        :基于OpenCV的自动驾驶平台模拟控制
@版本        :1.0
'''


# 导入系统库
import cv2
import numpy as np
import math
import gym
import gym_donkeycar

# 导入自定义库
from tools import region_of_interest,detect_line,make_points,average_lines,display_line


def main():
    '''
    主函数
    '''
    # 设置模拟器环境
    env = gym.make("donkey-generated-roads-v0")

    # 重置当前场景
    obv = env.reset()

    # 开始启动
    action = np.array([0, 0.3])  # 动作控制，第1个转向值，第2个油门值

    # 执行动作
    obv, reward, done, info = env.step(action)

    # 获取图像
    frame = cv2.cvtColor(obv, cv2.COLOR_RGB2BGR)

    # 运行5000次动作
    for t in range(5000):
        # 转换图像到HSV空间
        height, width, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 黄色区域检测
        lower_blue = np.array([15, 40, 40])
        upper_blue = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        # 白色区域检测
        lower_blue = np.array([0, 0, 200])
        upper_blue = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        # 黄色线边缘提取
        yellow_edge = cv2.Canny(yellow_mask, 200, 400)
        
        # 白色线边缘提取
        white_edge = cv2.Canny(white_mask, 200, 400)

        # 黄色线感兴趣区域提取
        yellow_roi = region_of_interest(yellow_edge, color='yellow')

        # 白色线感兴趣区域提取
        white_roi = region_of_interest(white_edge, color='white')

        # 黄色线段检测
        yellow_lines = detect_line(yellow_roi)
        yellow_lane = average_lines(frame, yellow_lines, direction='left')
        #yellow_show = display_line(frame, yellow_lane)

        # 白色线段检测
        white_lines = detect_line(white_roi)
        white_lane = average_lines(frame, white_lines, direction='right')
        #white_show = display_line(frame, white_lane, line_color=(255, 0, 0))

        # 计算转向角
        x_offset = 0
        y_offset = 0
        if len(yellow_lane)>0 and len(white_lane)>0:  # 检测到2条线
            _, _, left_x2, _ = yellow_lane[0][0]
            _, _, right_x2, _ = white_lane[0][0]
            mid = int(width / 2)
            x_offset = (left_x2 + right_x2) / 2 - mid
            y_offset = int(height / 2)
        elif len(yellow_lane)>0 and len(yellow_lane[0])==1:  # 只检测到黄色行道线
            x1, _, x2, _ = yellow_lane[0][0]
            x_offset = x2 - x1
            y_offset = int(height / 2)
        elif len(white_lane)>0 and len(white_lane[0])==1:  # 只检测到白色行道线
            x1, _, x2, _ = white_lane[0][0]
            x_offset = x2 - x1
            y_offset = int(height / 2)
        else: # 一条线都没检测到
            print('检测不到行道线，退出程序')
            break

        angle_to_mid_radian = math.atan(x_offset / y_offset)  
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi) 
        steering_angle = angle_to_mid_deg/45.0
        action = np.array([steering_angle, 0.1])  # 油门值恒定

        # 执行动作
        obv, reward, done, info = env.step(action)

        # 重新获取图像
        frame = cv2.cvtColor(obv, cv2.COLOR_RGB2BGR)

    # 运行完以后重置当前场景
    obv = env.reset()


if __name__ == '__main__':
    '''
    主函数入口
    '''
    main()
