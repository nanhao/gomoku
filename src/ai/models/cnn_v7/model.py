"""CNN v7 双头残差卷积神经网络模型实现

CNN v7 在 cnn_v6 残差网络基础上增加价值头，形成双头输出：
1. BatchNorm：每层 Conv 后添加 BN（Conv→BN→ReLU）
2. 残差连接：Conv2+Conv3 组成残差块1，Conv4+Conv5 组成残差块2
3. 策略头：独立 conv_out(32→1,1×1) → Flatten → LogSoftmax 输出 225 维落子概率
4. 价值头：独立 value_conv_out(32→1,1×1) → Flatten → ReLU → FC(225→1) → Tanh，输出 [-1,1] 胜率
   价值头在共享特征层（conv_out 之前）分支，参数量仅 259（相比旧版 ↓112×）

网络结构：
  输入 (batch, 3, 15, 15)
  → Conv1(3→32) → BN1 → ReLU1
  → [残差块1] Conv2→BN2→ReLU2→Conv3→BN3→[+shortcut]→ReLU3
  → [残差块2] Conv4→BN4→ReLU4→Conv5→BN5→[+shortcut]→ReLU5
  → 共享特征 (batch, 32, 15, 15)
      ├─ conv_out(32→1,1×1) → Flatten(225) → LogSoftmax → 策略 (batch, 225)
      └─ value_conv_out(32→1,1×1) → Flatten(225) → ReLU → FC(225→1) → Tanh → 价值 (batch, 1)

forward 返回 (policy, value) 元组。
predict 返回 ModelPredict 对象（type=policy_value）。
"""

import os
from typing import List

import torch
import torch.nn as nn

from ai.base_model import NeuralNetworkModel, board_vector_to_cnn_tensor_relative, ModelPredict


class CNNV7Network(nn.Module):
    """CNN v7 带 BN + 残差连接 + 轻量化价值头的双头卷积神经网络

    价值头在共享特征层独立分支，使用独立的 1×1 卷积 + 单层 FC，
    参数量仅 259（旧版 29,057），总参数量约 38,500。

    输入：(batch, 3, 15, 15) 棋盘状态
    输出：(policy, value) 元组
      - policy: (batch, 225) LogSoftmax log-probability
      - value:  (batch, 1) Tanh 胜率估值，值域 [-1, 1]
    """

    def __init__(self):
        """初始化 CNN v7 网络结构"""
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

        # ---- 策略头输出卷积：32→1, 1×1 ----
        self.conv_out = nn.Conv2d(32, 1, kernel_size=1)

        # ---- 策略头：LogSoftmax ----
        self.log_softmax = nn.LogSoftmax(dim=1)

        # ---- 价值头：独立的 1×1 卷积 + ReLU + FC(225→1) + Tanh ----
        self.value_conv_out = nn.Conv2d(32, 1, kernel_size=1)
        self.value_relu = nn.ReLU()
        self.value_fc = nn.Linear(225, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor):
        """前向传播

        Args:
            x: 输入张量 (batch, 3, 15, 15)

        Returns:
            (policy, value) 元组：
            - policy: (batch, 225) LogSoftmax log-probability
            - value:  (batch, 1) Tanh 胜率估值，值域 [-1, 1]
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

        # ---- 共享特征层：32 通道特征图，策略头与价值头在此分支 ----
        # x 形状: (batch, 32, 15, 15)

        # ---- 策略头：conv_out → Flatten → LogSoftmax ----
        policy = self.conv_out(x)           # (batch, 1, 15, 15)
        policy = policy.view(policy.size(0), -1)  # (batch, 225)
        policy = self.log_softmax(policy)   # (batch, 225)

        # ---- 价值头：value_conv_out → Flatten → ReLU → FC → Tanh ----
        value = self.value_conv_out(x)      # (batch, 1, 15, 15)
        value = value.view(value.size(0), -1)  # (batch, 225)
        value = self.value_relu(value)      # (batch, 225)
        value = self.value_fc(value)        # (batch, 1)
        value = self.value_tanh(value)      # (batch, 1)

        return policy, value


class CNNV7Model(NeuralNetworkModel):
    """CNN v7 双头残差卷积神经网络模型

    双头输出：策略（225 维落子概率） + 价值（[-1,1] 胜率估值）。
    继承 NeuralNetworkModel，通用逻辑由基类提供。
    predict 覆盖基类实现以返回 (策略, 价值) 元组。

    默认行为：
    - 如果 cnn_v7.pth 存在，自动加载该模型参数
    - 如果不存在，创建新的模型
    - 使用 fresh=True 可强制创建新模型
    """

    # 默认模型保存路径
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_v7.pth")

    @property
    def name(self) -> str:
        """模型名称"""
        return "cnn_v7"

    def _create_network(self) -> nn.Module:
        """创建 CNN v7 神经网络实例

        Returns:
            CNNV7Network 实例
        """
        return CNNV7Network()

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
        """预测落子概率和局面价值

        CNN v7 覆盖基类 predict，处理双头输出（策略 + 价值）。

        Args:
            board_vector: 226 维向量，前 225 位是棋盘（按行展开，0=空, 1=黑, 2=白），
                         最后 1 位是当前落子方（1=黑, 2=白）

        Returns:
            ModelPredict 对象，type=TYPE_POLICY_VALUE：
            - policy: 225 维落子概率列表，值域 (0, 1)，总和为 1
            - value: 价值标量，值域 [-1, 1]，从当前落子方视角评估胜率
        """
        device = next(self.network.parameters()).device
        input_tensor = self._prepare_input(board_vector, device)

        self.network.eval()
        with torch.no_grad():
            policy_tensor, value_tensor = self.network(input_tensor)

        # 策略：LogSoftmax → exp 还原为概率
        policy_probs = policy_tensor.squeeze(0).exp().tolist()
        policy_result = [float(p) for p in policy_probs]

        # 价值：标量
        value_result = float(value_tensor.squeeze().item())

        return ModelPredict(policy=policy_result, value=value_result)

    def predict_batch(self, board_vectors: List[List[int]]) -> List["ModelPredict"]:
        """批量预测落子概率和局面价值

        利用 CNNV7Network.forward 的 batch 输入能力，一次前向传播处理多个状态。

        Args:
            board_vectors: 多个 226 维向量列表

        Returns:
            等长 ModelPredict 列表
        """
        if not board_vectors:
            return []

        device = next(self.network.parameters()).device
        # 将多个向量拼接为 batch 张量 (B, 3, 15, 15)
        batch_tensors = [self._prepare_input(bv, device) for bv in board_vectors]
        input_tensor = torch.cat(batch_tensors, dim=0)

        self.network.eval()
        with torch.no_grad():
            policy_tensor, value_tensor = self.network(input_tensor)

        # 解析结果：LogSoftmax → exp 还原，逐个构建 ModelPredict
        results: List[ModelPredict] = []
        for i in range(len(board_vectors)):
            policy = [float(p) for p in policy_tensor[i].exp().tolist()]
            value = float(value_tensor[i].item())
            results.append(ModelPredict(policy=policy, value=value))
        return results
