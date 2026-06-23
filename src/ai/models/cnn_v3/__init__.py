"""CNN v3 极简卷积神经网络模型包

包含 CNN v3 极简模型实现，用于五子棋落子位置预测。
"""

from .model import CNNV3Model
from .trainer import CNNV3Trainer

__all__ = ["CNNV3Model", "CNNV3Trainer"]
