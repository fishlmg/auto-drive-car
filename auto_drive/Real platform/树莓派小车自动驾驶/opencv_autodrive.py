#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :opencv_auto_drive.py
@说明        :基于OpenCV的自动驾驶和数据采集
@时间        :2022/03/17 22:45:12
@作者        :钱彬
@版本        :1.0
'''

# 导入库函数
import cv2

# 导入小车运动库
from picarx import Picarx

# 导入自定义函数库
from tools import detect_lane, display_lines, compute_steer, stabilize_steering_angle, take_photo, ImageTrans

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

    # 循环控制
    while True:
        # 读取图像
        ret, frame = cap.read()
        if not ret:
            print("图像获取失败，请检查")
            break

        # 检测行道线
        lane_lines = detect_lane(frame)
        # lane_lines_image = display_lines(frame, lane_lines)
        # img_trans.sendImg(lane_lines_image)

        # 计算转向角
        line_num, steer_angle = compute_steer(lane_lines, frame)
        if line_num == 0:
            take_photo(-1, frame, pic_index)  # 记录异常图像
            pic_index += 1
            continue

        # 平滑处理
        last_steer_angle = stabilize_steering_angle(last_steer_angle,
                                                    steer_angle, line_num)

        # 输出转向值
        print("\r 当前转向角度steer: %s    " % (last_steer_angle))

        # 执行动作
        px.set_dir_servo_angle(last_steer_angle)
        px.forward(speed)

        # 保存图像
        if pic_index % 5 == 0:
            take_photo(last_steer_angle, frame, pic_index)
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