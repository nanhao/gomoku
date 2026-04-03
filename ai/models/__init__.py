"""AI 模型包

包含所有 AI 模型实现：
- random: 全随机 AI
- fc_v1: FC v1 全连接神经网络 AI
"""

from .random import RandomModel
from .fc_v1 import FCv1Model, FCv1Trainer

__all__ = [
    "RandomModel",
    "FCv1Model",
    "FCv1Trainer",
]
