"""CNN v10 价值头卷积神经网络模型包

在 cnn_v9 基础上去掉策略头，仅保留价值头。
输出单一局面价值评估，值域 [-1, 1]。
"""

from .model import CNNV10Model, CNNV10Network

__all__ = ["CNNV10Model", "CNNV10Network"]
