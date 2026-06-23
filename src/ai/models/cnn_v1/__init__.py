"""CNN 卷积神经网络模型包

包含 CNN v1 模型实现，用于五子棋落子位置预测。
"""

from .model import CNNV1Model
from .trainer import CNNV1Trainer

__all__ = ["CNNV1Model", "CNNV1Trainer"]
