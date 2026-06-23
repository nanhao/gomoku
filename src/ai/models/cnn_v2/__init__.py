"""CNN v2 轻量化卷积神经网络模型包

包含 CNN v2 模型实现，用于五子棋落子位置预测。
"""

from .model import CNNV2Model
from .trainer import CNNV2Trainer

__all__ = ["CNNV2Model", "CNNV2Trainer"]