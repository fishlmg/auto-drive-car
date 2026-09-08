#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :inference.py
@说明        :利用openvino进行单张图片推理
@时间        :2022/03/25 20:00:02
@作者        :钱彬
@版本        :1.0
'''


# 导入库
import cv2
import numpy as np
# 提前运行setupvars.bat脚本，保证环境变量已加载
from openvino.inference_engine import IECore


def main():
    '''
    主函数
    '''
    # 设置模型路径
    model = "./ir/autodrive.xml"
    
    # 设置推理环境（CPU、GPU、MYRIAD）
    device = 'MYRIAD'  # 'MYRIAD'

    # 设置推理引擎
    ie = IECore() 
    
    # 读取网络模型
    net = ie.read_network(model=model)

    # 设置推理图像
    input = 'test.jpg'
    
    # 获取模型输入形状
    for input_key in net.input_info:
        if len(net.input_info[input_key].input_data.layout) == 4:
            n, c, h, w = net.input_info[input_key].input_data.shape
    
    # 输入信息预处理
    images = np.ndarray(shape=(n, c, h, w))
    images_hw = []
    for i in range(n):
        image = cv2.imread(input)
        ih, iw = image.shape[:-1]
        images_hw.append((ih, iw))
        if (ih, iw) != (h, w):
            image = cv2.resize(image, (w, h))
        # BGR 转换为 HSV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # HWC 转换为 CHW 
        image = image.transpose((2, 0, 1))  
        # 转换至 (0,1)
        image = image / 255. 
        # # 减去均值
        # image -= np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
        # # 除以方差 
        # image /= np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        images[i] = image  
        
    assert (len(net.input_info.keys()) == 1 or len(
        net.input_info.keys()) == 2), "Sample supports topologies only with 1 or 2 inputs"
    out_blob = next(iter(net.outputs))
    # 单一输出，转向值
    for input_key in net.input_info:
        input_name = input_key
        net.input_info[input_key].precision = 'FP32'
        break
    
    data = {}
    data[input_name] = images

    # 准备执行网络
    exec_net = ie.load_network(network=net, device_name=device)
    
    # 推理
    res = exec_net.infer(inputs=data)
    
    # 输出结果
    for i, probs in enumerate(res[out_blob]):
        probs = np.squeeze(probs)
        print(probs)


if __name__=="__main__":
    '''
    程序入口
    '''
    main()