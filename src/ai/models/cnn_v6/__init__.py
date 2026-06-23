"""CNN v6 残差卷积神经网络模型包

在 cnn_v5 基础上引入 BatchNorm 和残差连接（Residual Block）。
纯策略网络，输出 225 维落子概率分布。
"""

from .model import CNNV6Model, CNNV6Network
from .trainer import CNNV6Trainer

__all__ = ["CNNV6Model", "CNNV6Network", "CNNV6Trainer"]
