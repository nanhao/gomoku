"""CNN v6 残差卷积神经网络模型实现

CNN v6 在 cnn_v5 基础上引入两个增强：
1. BatchNorm：每层 Conv 后添加 BN（Conv→BN→ReLU）
2. 残差连接：Conv2+Conv3 组成残差块1，Conv4+Conv5 组成残差块2

网络结构：
  输入 (batch, 3, 15, 15)
  → Conv1(3→32) → BN1 → ReLU1
  → [残差块1] Conv2→BN2→ReLU2→Conv3→BN3→[+shortcut]→ReLU3
  → [残差块2] Conv4→BN4→ReLU4→Conv5→BN5→[+shortcut]→ReLU5
  → Conv_out(32→1, 1×1) → Flatten→225
  → LogSoftmax → 策略 (batch, 225)

forward 返回 policy 张量。
predict 返回 List[float] 概率分布。
"""

import os
from typing import List

import torch
import torch.nn as nn

from ai.base_model import NeuralNetworkModel, board_vector_to_cnn_tensor, ModelPredict


class CNNV6Network(nn.Module):
    """CNN v6 带 BN + 残差连接的卷积神经网络

    在 cnn_v5 五层纯卷积基础上引入 BatchNorm 和残差连接。
    纯策略网络，输出 225 维落子 log-probability 分布。

    输入：(batch, 3, 15, 15) 棋盘状态
    输出：(batch, 225) LogSoftmax log-probability
    """

    def __init__(self):
        """初始化 CNN v6 网络结构"""
        super().__init__()

        # ---- 第一层：Conv1 + BN1 + ReLU1 ----
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        # ---- 残差块 1：Conv2 + Conv3 ----
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        # ---- 残差块 2：Conv4 + Conv5 ----
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        # ---- 输出卷积：32→1, 1×1 ----
        self.conv_out = nn.Conv2d(32, 1, kernel_size=1)

        # ---- 策略头：LogSoftmax ----
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor):
        """前向传播

        Args:
            x: 输入张量 (batch, 3, 15, 15)

        Returns:
            (batch, 225) LogSoftmax log-probability
        """
        # ---- 第一层：Conv1 + BN1 + ReLU1 ----
        x = self.relu1(self.bn1(self.conv1(x)))

        # ---- 残差块 1：Conv2 + BN2 + ReLU2 + Conv3 + BN3 + shortcut + ReLU3 ----
        shortcut = x
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + shortcut
        x = self.relu3(x)

        # ---- 残差块 2：Conv4 + BN4 + ReLU4 + Conv5 + BN5 + shortcut + ReLU5 ----
        shortcut = x
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x = x + shortcut
        x = self.relu5(x)

        # ---- 输出卷积 + Flatten ----
        x = self.conv_out(x)                # (batch, 1, 15, 15)
        x = x.view(x.size(0), -1)           # (batch, 225)

        # ---- 策略头：LogSoftmax ----
        x = self.log_softmax(x)             # (batch, 225)

        return x


class CNNV6Model(NeuralNetworkModel):
    """CNN v6 残差卷积神经网络模型

    纯策略网络，输出 225 维落子概率分布。
    继承 NeuralNetworkModel，通用逻辑由基类提供。

    默认行为：
    - 如果 cnn_v6.pth 存在，自动加载该模型参数
    - 如果不存在，创建新的模型
    - 使用 fresh=True 可强制创建新模型
    """

    # 默认模型保存路径
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_v6.pth")

    @property
    def name(self) -> str:
        """模型名称"""
        return "cnn_v6"

    def _create_network(self) -> nn.Module:
        """创建 CNN v6 神经网络实例

        Returns:
            CNNV6Network 实例
        """
        return CNNV6Network()

    def _prepare_input(self, board_vector: List[int], device: torch.device) -> torch.Tensor:
        """将 226 维向量转换为 CNN 输入张量 (1, 3, 15, 15)

        Args:
            board_vector: 226 维向量
            device: 目标设备

        Returns:
            形状为 (1, 3, 15, 15) 的张量
        """
        return board_vector_to_cnn_tensor(board_vector, device)

    def predict(self, board_vector: List[int]) -> "ModelPredict":
        """预测落子概率

        Args:
            board_vector: 226 维向量，前 225 位是棋盘（按行展开，0=空, 1=黑, 2=白），
                         最后 1 位是当前落子方（1=黑, 2=白）

        Returns:
            ModelPredict 对象，type=TYPE_POLICY_ONLY，value=None
        """
        device = next(self.network.parameters()).device
        input_tensor = self._prepare_input(board_vector, device)

        self.network.eval()
        with torch.no_grad():
            policy_tensor = self.network(input_tensor)

        # 策略：LogSoftmax → exp 还原为概率
        policy_probs = policy_tensor.squeeze(0).exp().tolist()
        return ModelPredict(policy=[float(p) for p in policy_probs])
