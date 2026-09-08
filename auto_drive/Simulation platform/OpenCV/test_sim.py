#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :test_sim.py
@说明        :测试小车控制
@时间        :2022/03/02 14:18:50
@作者        :钱彬
@版本        :1.0
'''

# 导入库
import gym
import gym_donkeycar
import numpy as np
import cv2
import math

# 设置模拟器环境
env = gym.make("donkey-generated-roads-v0")

# 重置当前场景
obv = env.reset()

steel = 0
while True:
    action = np.array([steel, 0.1])  # 动作控制:转向+油门
    print(steel)
    # 执行动作
    img, reward, done, info = env.step(action)
    cv2.imshow('capture', img)

    k = cv2.waitKey(3) & 0xFF
    if k == ord('a'):  # 左转
        if steel <= -0.3:
            steel -= 0.02
        elif steel <= -0.15:
            steel -= 0.05
        else:
            steel -= 0.15
    elif k == ord('d'):  # 右转
        if steel >= 0.3:
            steel += 0.02
        if steel >= 0.15:
            steel += 0.05
        else:
            steel += 0.15
    elif k == ord('q'):  # 退出结束
        break
    else:  # 什么键都不按方向逐步归正
        if steel > 0.04:
            steel -= 0.04
        elif steel < -0.04:
            steel += 0.04
        else:
            steel = 0

    # 最后矫正
    thr = 0.6
    if steel > thr:
        steel = thr
    if steel < -thr:
        steel = -thr

# 运行完以后重置当前场景
print('结束')
obv = env.reset()