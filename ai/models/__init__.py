"""AI 模型包

包含所有 AI 模型实现：
- random: 全随机 AI
- fc_v1: FC v1 全连接神经网络 AI
- cnn_v1: CNN v1 卷积神经网络 AI
- cnn_v2: CNN v2 轻量化卷积神经网络 AI
"""

from .random import RandomModel
from .fc_v1 import FCv1Model, FCv1Trainer
from .cnn_v1 import CNNV1Model, CNNV1Trainer
from .cnn_v2 import CNNV2Model, CNNV2Trainer

__all__ = [
    "RandomModel",
    "FCv1Model",
    "FCv1Trainer",
    "CNNV1Model",
    "CNNV1Trainer",
    "CNNV2Model",
    "CNNV2Trainer",
]
