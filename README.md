# 基于深度学习的自动驾驶小车

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

自动驾驶小车项目，覆盖**仿真平台**（Donkey Car 仿真器）与**实物平台**（树莓派 PiCar-X），实现两条技术路线：基于 OpenCV 的传统车道线检测，以及基于 PyTorch CNN 的行为克隆（Behavioral Cloning）端到端驾驶。

## 🏗 双平台 × 双路线

| | OpenCV 车道线 | 深度学习行为克隆 |
| --- | --- | --- |
| **仿真平台** | ✅ `Simulation platform/OpenCV/` | ✅ `Simulation platform/deep_learning/` |
| **实物平台** | ✅ `Real platform/树莓派小车自动驾驶/` | ✅ `Real platform/Pytorch深度学习训练推理/` |

## ✨ 特性

- **OpenCV 车道线方案**：ROI 提取 → Canny 边缘 → 霍夫直线 → 车道线拟合 → 转向角计算，带转向平滑稳定
- **行为克隆方案**：采集（图像, 转向角）数据 → CNN 回归转向角 → 导出 ONNX 实车部署
- **仿真到实车**：Donkey Car 仿真器中采集数据、训练模型，再迁移到树莓派实车
- **摄像头标定**：实车部署前的相机标定工具
- **模型导出**：支持导出 ONNX / OpenVINO IR，适配低算力设备

## 🗂 项目结构

```
auto_drive/
├── Real platform/                          # 实物平台（树莓派 PiCar-X）
│   ├── Pytorch深度学习训练推理/              # 行为克隆模型
│   │   ├── train.py / eval.py              # 训练与评估
│   │   ├── models.py                       # CNN 网络结构
│   │   ├── datasets.py                     # 数据集加载
│   │   ├── create_data_lists.py           # 数据清单生成
│   │   ├── export_onnx.py                  # 导出 ONNX
│   │   └── utils.py
│   └── 树莓派小车自动驾驶/                   # 实车部署
│       ├── opencv_autodrive.py             # OpenCV 车道线自动驾驶
│       ├── deeplearning_autodrive.py       # 深度学习模型自动驾驶
│       ├── inference.py                    # 模型推理
│       ├── move.py                         # 小车运动控制
│       ├── tools.py                        # 车道线检测工具函数
│       ├── video.py                        # 视频采集
│       └── calibration.py                  # 摄像头标定
│
└── Simulation platform/                    # 仿真平台（Donkey Car）
    ├── deep_learning/                      # 仿真数据训练行为克隆
    │   ├── auto_drive.py                   # 仿真自动驾驶主程序
    │   ├── train.py / eval.py
    │   ├── models.py / datasets.py
    │   └── data/                           # 仿真采集数据（不入库）
    └── OpenCV/
        ├── auto_drive.py / test_sim.py     # 仿真车道线驾驶
        ├── collect_data.py                 # 仿真驾驶数据采集
        └── tools.py
```

## 🚀 快速开始

```bash
pip install -r requirements.txt
```

**路线一：行为克隆模型训练**

```bash
# 1. 采集数据（仿真器或实车运行 collect_data.py / 实车录制）
# 2. 生成数据清单
cd "Real platform/Pytorch深度学习训练推理"
python create_data_lists.py
# 3. 训练
python train.py
# 4. 评估并导出
python eval.py
python export_onnx.py
```

**路线二：OpenCV 车道线（实车）**

```bash
cd "Real platform/树莓派小车自动驾驶"
python calibration.py          # 首次部署先标定摄像头
python opencv_autodrive.py     # 启动车道线自动驾驶
```

## 🧠 技术原理

**OpenCV 车道线：**

1. 图像转灰度 + 高斯模糊 + Canny 边缘检测
2. ROI 掩码提取车道区域
3. 霍夫变换检测直线 → 拟合左右车道线
4. 根据车道线斜率/截距计算转向角
5. 指数平滑稳定转向，避免抖动

**行为克隆（端到端）：**

1. 人驾驶采集（前置图像, 转向角度）样本对
2. CNN 以图像为输入、回归输出转向角
3. 训练后模型直接输出转向指令，无需人工设计视觉特征

## 📦 依赖

PyTorch、torchvision、OpenCV、NumPy、Pillow、TensorBoard、onnx、gym、gym-donkeycar；实车还需 `picarx`、`readchar`（详见 [requirements.txt](requirements.txt) 注释）

## 📝 说明

- Unity 仿真器二进制（`Simulation platform controller/`）、采集的图像数据集、模型权重、视频与压缩包不包含在仓库中（`.gitignore` 已排除）
- 树莓派硬件相关库（picarx / openvino / picamera）仅在实车环境需要

## 📄 许可证

[MIT License](LICENSE)
