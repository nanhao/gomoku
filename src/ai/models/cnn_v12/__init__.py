"""CNN v12 轻量双头残差卷积神经网络模型包

cnn_v11 的等比例瘦身版本：128 通道 + 4 残差块 + AGZ 风格策略/价值头。
参数量 1,346,541（cnn_v11 的 16.0%）。

依赖声明：本包依赖 ai.models.cnn_v11
（复用 ResBlock / CNNV11Model / CNNV11Trainer）。
若删除 cnn_v11 包，cnn_v12 将同时失效。
"""

from .model import CNNV12Model, CNNV12Network
from .trainer import CNNV12Trainer

__all__ = ["CNNV12Model", "CNNV12Network", "CNNV12Trainer"]
