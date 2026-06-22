"""CNN v6 残差卷积神经网络模型包

在 cnn_v5 基础上引入 BatchNorm 和残差连接（Residual Block），
并增加价值头（Value Head）输出当前局面胜率估值。
"""

from .model import CNNV6Model, CNNV6Network
from .trainer import CNNV6Trainer

__all__ = ["CNNV6Model", "CNNV6Network", "CNNV6Trainer"]
