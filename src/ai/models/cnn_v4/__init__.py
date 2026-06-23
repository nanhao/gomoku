"""CNN v4 卷积神经网络模型包

在 cnn_v3 基础上增加一个 32 通道隐藏层，保持纯卷积结构。
"""

from .model import CNNV4Model
from .trainer import CNNV4Trainer

__all__ = ["CNNV4Model", "CNNV4Trainer"]
