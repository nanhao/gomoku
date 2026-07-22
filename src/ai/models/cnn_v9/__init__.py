"""CNN v9 双头残差卷积神经网络模型包

在 cnn_v8 基础上将所有卷积层通道数从 32 提升到 48，
提升特征提取能力。参数量约 128,300。
"""

from .model import CNNV9Model, CNNV9Network
from .trainer import CNNV9Trainer

__all__ = ["CNNV9Model", "CNNV9Network", "CNNV9Trainer"]
