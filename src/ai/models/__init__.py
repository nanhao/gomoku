"""AI 模型包

包含所有 AI 模型实现：
- random: 全随机 AI
- fc_v1: FC v1 全连接神经网络 AI
- cnn_v1: CNN v1 卷积神经网络 AI
- cnn_v2: CNN v2 轻量化卷积神经网络 AI
- cnn_v3: CNN v3 极简卷积神经网络 AI
- cnn_v4: CNN v4 卷积神经网络 AI（cnn_v3 + 额外隐藏层）
- cnn_v5: CNN v5 卷积神经网络 AI（cnn_v4 + 额外隐藏层）
- cnn_v6: CNN v6 残差卷积神经网络 AI（策略网络）
- cnn_v7: CNN v7 双头残差卷积神经网络 AI（策略 + 价值）
- cnn_v8: CNN v8 双头残差卷积神经网络 AI（cnn_v7 + FC 隐藏层）
"""

from .random import RandomModel
from .fc_v1 import FCv1Model, FCv1Trainer
from .cnn_v1 import CNNV1Model, CNNV1Trainer
from .cnn_v2 import CNNV2Model, CNNV2Trainer
from .cnn_v3 import CNNV3Model, CNNV3Trainer
from .cnn_v4 import CNNV4Model, CNNV4Trainer
from .cnn_v5 import CNNV5Model, CNNV5Trainer
from .cnn_v6 import CNNV6Model, CNNV6Trainer
from .cnn_v7 import CNNV7Model, CNNV7Trainer
from .cnn_v8 import CNNV8Model, CNNV8Trainer

__all__ = [
    "RandomModel",
    "FCv1Model",
    "FCv1Trainer",
    "CNNV1Model",
    "CNNV1Trainer",
    "CNNV2Model",
    "CNNV2Trainer",
    "CNNV3Model",
    "CNNV3Trainer",
    "CNNV4Model",
    "CNNV4Trainer",
    "CNNV5Model",
    "CNNV5Trainer",
    "CNNV6Model",
    "CNNV6Trainer",
    "CNNV7Model",
    "CNNV7Trainer",
    "CNNV8Model",
    "CNNV8Trainer",
]
