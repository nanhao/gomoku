"""CNN v5 卷积神经网络模型包

在 cnn_v4 基础上增加一个 32 通道隐藏层，保持纯卷积结构。
"""

from .model import CNNV5Model
from .trainer import CNNV5Trainer

__all__ = ["CNNV5Model", "CNNV5Trainer"]
