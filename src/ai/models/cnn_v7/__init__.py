"""CNN v7 双头残差卷积神经网络模型包

在 cnn_v6 残差网络基础上增加价值头（Value Head），
形成策略+价值的双头输出架构。
"""

from .model import CNNV7Model, CNNV7Network
from .trainer import CNNV7Trainer

__all__ = ["CNNV7Model", "CNNV7Network", "CNNV7Trainer"]
