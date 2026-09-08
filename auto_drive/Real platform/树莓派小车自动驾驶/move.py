#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :move.py
@说明        :树莓派小车基本运动控制
@时间        :2022/03/25 21:29:12
@作者        :钱彬
@版本        :1.0
'''

from picarx import Picarx
import time


def main():
    '''
    主程序，基本程序控制测试
    '''
    px = Picarx()
    try:
        while True:
            # 前进
            px.forward(5)
            time.sleep(0.5)
            # 向右摆头
            for angle in range(0, 35):
                px.set_dir_servo_angle(angle)
                time.sleep(0.02)
            # 从右向左摆头
            for angle in range(35, -35, -1):
                px.set_dir_servo_angle(angle)
                time.sleep(0.02)
            # 从左向右摆头至中间位
            for angle in range(-35, 0):
                px.set_dir_servo_angle(angle)
                time.sleep(0.02)
    except KeyboardInterrupt:
        px.forward(0)
        px.set_dir_servo_angle(0)


if __name__ == '__main__':
    '''
    程序入口
    '''
    main()
