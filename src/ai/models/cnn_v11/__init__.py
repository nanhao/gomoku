"""CNN v11 双头残差卷积神经网络模型包

256 通道 + 7 残差块 + AGZ 风格策略/价值头。
参数量约 843 万（cnn_v9 的 66 倍）。
"""

from .model import CNNV11Model, CNNV11Network
from .trainer import CNNV11Trainer

__all__ = ["CNNV11Model", "CNNV11Network", "CNNV11Trainer"]
