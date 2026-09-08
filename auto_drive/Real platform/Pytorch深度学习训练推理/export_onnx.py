#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :export_onnx.py
@说明        :导出onnx文件
@时间        :2022/03/23 21:13:19
@作者        :钱彬
@版本        :1.0
'''


# 导入OpenCV库
import cv2

# 导入PyTorch库
from torch import nn
import torch
import torch.onnx

# 导入onnx库
import onnx

# 导入自定义库
from models import AutoDriveNet
from utils import *


def main():
    '''
    主函数
    '''
    # 测试图像
    imgPath = './results/test.jpg'

    # 推理环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型
    checkpoint = torch.load('./results/checkpoint.pth')
    model = AutoDriveNet()
    model = model.to(device)
    model.load_state_dict(checkpoint['model'],strict=False)

    # 加载图像
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (160,120))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 图像预处理
    # PIXEL_MEANS = (0.485, 0.456, 0.406)  # RGB格式的均值和方差
    # PIXEL_STDS = (0.229, 0.224, 0.225)
    img = torch.from_numpy(img.copy()).float()
    img /= 255.0
    # img -= torch.tensor(PIXEL_MEANS)
    # img /= torch.tensor(PIXEL_STDS)
    img = img.permute(2, 0, 1)
    img.unsqueeze_(0)

    # 转移数据至设备
    img = img.to(device)

    # 模型推理
    model.eval()
    with torch.no_grad():
        prelabel = model(img).squeeze(0).cpu().detach().numpy()
        torch.onnx.export(model,                     # model being run
                  img,                         # model input (or a tuple for multiple inputs)
                  "results/autodrive.onnx",       # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
        print('预测结果  {:.3f} '.format(prelabel[0]))


if __name__ == '__main__':
    '''
    程序入口
    '''
    main()