"""CNN v10 价值头残差卷积神经网络模型实现

CNN v10 在 cnn_v9 基础上去掉策略头，仅保留价值头：
1. 共享特征层：与 cnn_v9 完全一致（Conv1-5 + BN + 残差连接，通道 48）
2. 价值头：value_conv_out(48→1,1×1) → Flatten(225) → FC(225→64) → ReLU → FC(64→1) → Tanh

网络结构：
  输入 (batch, 3, 15, 15)
  → Conv1(3→48) → BN1 → ReLU1
  → [残差块1] Conv2→BN2→ReLU2→Conv3→BN3→[+shortcut]→ReLU3
  → [残差块2] Conv4→BN4→ReLU4→Conv5→BN5→[+shortcut]→ReLU5
  → 共享特征 (batch, 48, 15, 15)
  → value_conv_out(48→1,1×1) → Flatten(225) → FC(225→64) → ReLU → FC(64→1) → Tanh
  → 价值 (batch, 1)

forward 返回单个 (batch, 1) 张量，值域 [-1, 1]。
predict 返回 ModelPredict 对象（type=value_only）。

参数量：约 70,000
"""

import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.base_model import NeuralNetworkModel, board_vector_to_cnn_tensor_relative, ModelPredict


class CNNV10Network(nn.Module):
    """CNN v10 带 BN + 残差连接的价值头卷积神经网络

    在 CNNV9Network 基础上删除策略头全部组件，
    仅保留共享特征层和价值头。

    输入：(batch, 3, 15, 15) 棋盘状态
    输出：(batch, 1) Tanh 胜率估值，值域 [-1, 1]
    """

    def __init__(self, hidden_size: int = 64):
        """初始化 CNN v10 网络结构

        Args:
            hidden_size: FC 隐藏层维度，默认 64
        """
        super().__init__()
        self.hidden_size = hidden_size

        # ---- 第一层：Conv1(3→48) + BN1 + ReLU1 ----
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu1 = nn.ReLU()

        # ---- 残差块 1：Conv2(48→48) + Conv3(48→48) ----
        self.conv2 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.relu3 = nn.ReLU()

        # ---- 残差块 2：Conv4(48→48) + Conv5(48→48) ----
        self.conv4 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(48)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(48)
        self.relu5 = nn.ReLU()

        # ---- 价值头：独立 1×1 卷积(48→1) + FC 隐藏层 + Tanh ----
        self.value_conv_out = nn.Conv2d(48, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(225, hidden_size)
        self.value_relu1 = nn.ReLU()
        self.value_fc2 = nn.Linear(hidden_size, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 (batch, 3, 15, 15)

        Returns:
            价值张量 (batch, 1)，值域 [-1, 1]
        """
        # ---- 第一层：Conv1 + BN1 + ReLU1 ----
        x = self.relu1(self.bn1(self.conv1(x)))

        # ---- 残差块 1 ----
        shortcut = x
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + shortcut
        x = self.relu3(x)

        # ---- 残差块 2 ----
        shortcut = x
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x = x + shortcut
        x = self.relu5(x)

        # ---- 共享特征层：48 通道特征图 ----
        # x 形状: (batch, 48, 15, 15)

        # ---- 价值头：value_conv_out → Flatten → FC → ReLU → FC → Tanh ----
        value = self.value_conv_out(x)                  # (batch, 1, 15, 15)
        value = value.view(value.size(0), -1)            # (batch, 225)
        value = self.value_relu1(self.value_fc1(value))  # (batch, hidden_size)
        value = self.value_fc2(value)                    # (batch, 1)
        value = self.value_tanh(value)                   # (batch, 1)

        return value


class CNNV10Model(NeuralNetworkModel):
    """CNN v10 价值头残差卷积神经网络模型

    单输出：价值（[-1,1] 胜率估值）。
    在 cnn_v9 基础上去掉策略头。

    默认行为：
    - 如果 cnn_v10.pth 存在，自动加载该模型参数
    - 如果不存在，创建新的模型
    - 使用 fresh=True 可强制创建新模型
    """

    # 默认模型保存路径
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_v10.pth")

    @property
    def name(self) -> str:
        """模型名称"""
        return "cnn_v10"

    def _create_network(self) -> nn.Module:
        """创建 CNN v10 神经网络实例

        Returns:
            CNNV10Network 实例
        """
        return CNNV10Network()

    def _prepare_input(self, board_vector: List[int], device: torch.device) -> torch.Tensor:
        """将 226 维向量转换为相对视角 CNN 输入张量 (1, 3, 15, 15)

        使用相对视角编码（AlphaGo Zero 风格）：
        - 通道0：我方棋子
        - 通道1：对方棋子
        - 通道2：我方颜色指示器

        Args:
            board_vector: 226 维向量
            device: 目标设备

        Returns:
            形状为 (1, 3, 15, 15) 的张量
        """
        return board_vector_to_cnn_tensor_relative(board_vector, device)

    def predict(self, board_vector: List[int]) -> "ModelPredict":
        """预测局面价值

        Args:
            board_vector: 226 维向量，前 225 位是棋盘（按行展开，0=空, 1=黑, 2=白），
                         最后 1 位是当前落子方（1=黑, 2=白）

        Returns:
            ModelPredict 对象，type=TYPE_VALUE_ONLY：
            - value: 价值标量，值域 [-1, 1]，从当前落子方视角评估胜率
        """
        device = next(self.network.parameters()).device
        input_tensor = self._prepare_input(board_vector, device)

        self.network.eval()
        with torch.no_grad():
            value_tensor = self.network(input_tensor)

        # 价值：标量
        value_result = float(value_tensor.squeeze().item())

        return ModelPredict(value=value_result)

    def predict_batch(self, board_vectors: List[List[int]]) -> List["ModelPredict"]:
        """批量预测局面价值

        Args:
            board_vectors: 多个 226 维向量列表

        Returns:
            等长 ModelPredict 列表，每个 type=TYPE_VALUE_ONLY
        """
        if not board_vectors:
            return []

        device = next(self.network.parameters()).device
        batch_tensors = [self._prepare_input(bv, device) for bv in board_vectors]
        input_tensor = torch.cat(batch_tensors, dim=0)

        self.network.eval()
        with torch.no_grad():
            value_tensor = self.network(input_tensor)

        results: List[ModelPredict] = []
        for i in range(len(board_vectors)):
            value = float(value_tensor[i].item())
            results.append(ModelPredict(value=value))
        return results
