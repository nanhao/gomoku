"""CNN v8 双头残差卷积神经网络模型包

在 cnn_v7 基础上为双头各添加一层 FC 隐藏层，
提升模型容量以处理 AGZ 从零训练的噪声。参数量约 82,000。
"""

from .model import CNNV8Model, CNNV8Network
from .trainer import CNNV8Trainer

__all__ = ["CNNV8Model", "CNNV8Network", "CNNV8Trainer"]
