"""CNN v11 双头残差卷积神经网络模型实现

CNN v11 在 cnn_v9 基础上大幅增强：256 通道、7 个残差块、AGZ 风格策略头。
参照 AlphaGo Zero 论文网络结构：
1. 共享卷积体：1 层初始 Conv + 7×2=15 层 Conv，全部 256 通道
2. 策略头：conv_out(256→2,1×1) → BN → ReLU → Flatten(450) → FC(450→225) → LogSoftmax
3. 价值头：value_conv_out(256→1,1×1) → BN → ReLU → Flatten(225) → FC(225→256) → ReLU → FC(256→1) → Tanh

网络结构：
  输入 (batch, 3, 15, 15)
  → Conv1(3→256) → BN1 → ReLU1
  → ResBlock × 7（每块 Conv→BN→ReLU→Conv→BN→+shortcut→ReLU）
  → 共享特征 (batch, 256, 15, 15)
      ├─ conv_out(256→2,1×1) → BN → ReLU → Flatten(450) → FC(450→225) → LogSoftmax → 策略
      └─ value_conv_out(256→1,1×1) → BN → ReLU → Flatten(225) → FC(225→256) → ReLU → FC(256→1) → Tanh → 价值

forward 返回 (policy, value) 元组。
predict 返回 ModelPredict 对象（type=policy_value）。

参数量：约 843 万（cnn_v9 的 66 倍）
"""

import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.base_model import NeuralNetworkModel, board_vector_to_cnn_tensor_relative, ModelPredict


class ResBlock(nn.Module):
    """256 通道残差块

    每个残差块包含两层 Conv2d(256→256, 3×3, p=1) + BN + ReLU + shortcut。

    结构：
        x → Conv2d → BN → ReLU → Conv2d → BN → [+ identity] → ReLU → out
    """

    def __init__(self, channels: int = 256):
        """初始化残差块

        Args:
            channels: 卷积通道数，默认 256
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """残差块前向传播

        Args:
            x: 输入张量 (batch, channels, H, W)

        Returns:
            同形状输出张量
        """
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu2(out)
        return out


class CNNV11Network(nn.Module):
    """CNN v11 双头残差卷积神经网络

    256 通道 + 7 残差块 + AGZ 风格策略/价值头。

    输入：(batch, 3, 15, 15) 棋盘状态
    输出：(policy, value) 元组
      - policy: (batch, 225) LogSoftmax log-probability
      - value:  (batch, 1) Tanh 胜率估值，值域 [-1, 1]
    """

    def __init__(self):
        """初始化 CNN v11 网络结构"""
        super().__init__()

        # ---- 第一层：Conv1(3→256) + BN1 + ReLU1 ----
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

        # ---- 7 个残差块（ModuleList） ----
        self.res_blocks = nn.ModuleList([ResBlock(256) for _ in range(7)])

        # ---- 策略头：1×1 卷积(256→2) + BN + ReLU + Flatten + FC(450→225) + LogSoftmax ----
        self.conv_out = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU()
        self.policy_fc = nn.Linear(450, 225)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # ---- 价值头：1×1 卷积(256→1) + BN + ReLU + Flatten + FC(225→256) + ReLU + FC(256→1) + Tanh ----
        self.value_conv_out = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu1 = nn.ReLU()
        self.value_fc1 = nn.Linear(225, 256)
        self.value_relu2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # ---- 7 个残差块 ----
        for block in self.res_blocks:
            x = block(x)

        # ---- 共享特征层：256 通道特征图 ----
        # x 形状: (batch, 256, 15, 15)

        # ---- 策略头：conv_out → BN → ReLU → Flatten → FC → LogSoftmax ----
        policy = self.policy_relu(self.policy_bn(self.conv_out(x)))   # (batch, 2, 15, 15)
        policy = policy.view(policy.size(0), -1)                       # (batch, 450)
        policy = self.policy_fc(policy)                                # (batch, 225)
        policy = self.log_softmax(policy)                              # (batch, 225)

        # ---- 价值头：value_conv_out → BN → ReLU → Flatten → FC → ReLU → FC → Tanh ----
        value = self.value_relu1(self.value_bn(self.value_conv_out(x)))  # (batch, 1, 15, 15)
        value = value.view(value.size(0), -1)                             # (batch, 225)
        value = self.value_fc1(value)                                     # (batch, 256)
        value = self.value_relu2(value)                                   # (batch, 256)
        value = self.value_fc2(value)                                     # (batch, 1)
        value = self.value_tanh(value)                                    # (batch, 1)

        return policy, value


class CNNV11Model(NeuralNetworkModel):
    """CNN v11 双头残差卷积神经网络模型

    双头输出：策略（225 维落子概率） + 价值（[-1,1] 胜率估值）。

    默认行为：
    - 如果 cnn_v11.pth 存在，自动加载该模型参数
    - 如果不存在，创建新的模型
    - 使用 fresh=True 可强制创建新模型
    """

    # 默认模型保存路径
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_v11.pth")

    @property
    def name(self) -> str:
        """模型名称"""
        return "cnn_v11"

    def _create_network(self) -> nn.Module:
        """创建 CNN v11 神经网络实例

        Returns:
            CNNV11Network 实例
        """
        return CNNV11Network()

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

        Args:
            board_vectors: 多个 226 维向量列表

        Returns:
            等长 ModelPredict 列表
        """
        if not board_vectors:
            return []

        device = next(self.network.parameters()).device
        batch_tensors = [self._prepare_input(bv, device) for bv in board_vectors]
        input_tensor = torch.cat(batch_tensors, dim=0)

        self.network.eval()
        with torch.no_grad():
            policy_tensor, value_tensor = self.network(input_tensor)

        results: List[ModelPredict] = []
        for i in range(len(board_vectors)):
            policy = [float(p) for p in policy_tensor[i].exp().tolist()]
            value = float(value_tensor[i].item())
            results.append(ModelPredict(policy=policy, value=value))
        return results
