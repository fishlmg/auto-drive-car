# 基于深度学习的自动驾驶小车

自动驾驶小车项目，包含**仿真平台**（Donkey Car 仿真器）与**实物平台**（树莓派 PiCar-X）两部分，实现车道线检测（OpenCV）与行为克隆端到端驾驶（PyTorch）两条技术路线。

## 目录结构

```
auto_drive/
├── Real platform/                      # 实物平台
│   ├── Pytorch深度学习训练推理/          # 行为克隆模型
│   │   ├── train.py / eval.py          # 训练与评估
│   │   ├── models.py / datasets.py     # 网络结构与数据加载
│   │   ├── export_onnx.py              # 导出 ONNX
│   │   └── results/                    # 训练结果
│   └── 树莓派小车自动驾驶/               # 实车部署
│       ├── opencv_autodrive.py         # OpenCV 车道线自动驾驶
│       ├── deeplearning_autodrive.py   # 深度学习模型自动驾驶
│       ├── move.py / tools.py          # 小车控制与工具函数
│       └── calibration.py              # 摄像头标定
└── Simulation platform/                # 仿真平台
    ├── deep_learning/                  # 仿真数据训练行为克隆模型
    └── OpenCV/                         # 仿真环境车道线检测
        ├── audo_drive.py / test_sim.py
        └── collect_data.py             # 仿真数据采集
```

## 技术路线

1. **OpenCV 车道线**：ROI 提取 → 边缘检测 → 霍夫直线 → 拟合车道线 → 计算转向角
2. **深度学习行为克隆**：采集（图像, 转向角）数据 → CNN 回归转向角 → 导出 ONNX 部署

## 快速开始

```bash
pip install -r requirements.txt

# 训练行为克隆模型
cd "auto_drive/Real platform/Pytorch深度学习训练推理"
python create_data_lists.py
python train.py

# OpenCV 车道线（实车）
cd "auto_drive/Real platform/树莓派小车自动驾驶"
python opencv_autodrive.py
```

## 说明

- Unity 仿真器二进制（`Simulation platform controller/`）、压缩包、视频与模型权重不包含在仓库中（`.gitignore` 已排除）
- 树莓派硬件相关依赖（`picarx`、`readchar`、`openvino` 等）见 `requirements.txt` 注释，仅在实车环境需要
