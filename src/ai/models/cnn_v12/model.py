"""CNN v12 轻量双头残差卷积神经网络模型实现

CNN v12 是 cnn_v11 的等比例瘦身版本（设计 61）：
- 残差塔：7 块 → 4 块
- 卷积通道：256 → 128
策略头与价值头的结构与 cnn_v11 完全一致，仅 1×1 卷积的输入通道数随之减半。

依赖声明：本模块依赖 ai.models.cnn_v11.model
（复用 ResBlock 与 CNNV11Model）。若删除 cnn_v11 包，cnn_v12 将同时失效。

网络结构：
  输入 (batch, 3, 15, 15)
  → Conv1(3→128) → BN1 → ReLU1
  → ResBlock × 4（每块 Conv→BN→ReLU→Conv→BN→+shortcut→ReLU）
  → 共享特征 (batch, 128, 15, 15)
      ├─ conv_out(128→2,1×1) → BN → ReLU → Flatten(450) → FC(450→225) → LogSoftmax → 策略
      └─ value_conv_out(128→1,1×1) → BN → ReLU → Flatten(225) → FC(225→256) → ReLU → FC(256→1) → Tanh → 价值

forward 返回 (policy, value) 元组。
predict 返回 ModelPredict 对象（type=policy_value）。

参数量：1,346,541（cnn_v11 的 16.0%）
"""

import os
from typing import Tuple

import torch
import torch.nn as nn

from ai.models.cnn_v11.model import CNNV11Model, ResBlock


class CNNV12Network(nn.Module):
    """CNN v12 轻量双头残差卷积神经网络

    128 通道 + 4 残差块 + AGZ 风格策略/价值头。
    架构固定，不提供超参。

    输入：(batch, 3, 15, 15) 棋盘状态
    输出：(policy, value) 元组
      - policy: (batch, 225) LogSoftmax log-probability
      - value:  (batch, 1) Tanh 胜率估值，值域 [-1, 1]
    """

    # 全部卷积通道数（cnn_v11 的 256 减半为 128）
    CHANNELS: int = 128

    # 残差块个数（cnn_v11 的 7 块减为 4 块）
    NUM_RES_BLOCKS: int = 4

    def __init__(self):
        """初始化 CNN v12 网络结构"""
        super().__init__()

        channels = self.CHANNELS

        # ---- 第一层：Conv1(3→128) + BN1 + ReLU1 ----
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()

        # ---- 4 个残差块（复用 cnn_v11 的 ResBlock，该类已按 channels 参数化） ----
        self.res_blocks = nn.ModuleList(
            [ResBlock(channels) for _ in range(self.NUM_RES_BLOCKS)])

        # ---- 策略头：1×1 卷积(128→2) + BN + ReLU + Flatten + FC(450→225) + LogSoftmax ----
        self.conv_out = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU()
        self.policy_fc = nn.Linear(450, 225)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # ---- 价值头：1×1 卷积(128→1) + BN + ReLU + Flatten + FC(225→256) + ReLU + FC(256→1) + Tanh ----
        self.value_conv_out = nn.Conv2d(channels, 1, kernel_size=1)
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

        # ---- 4 个残差块 ----
        for block in self.res_blocks:
            x = block(x)

        # ---- 共享特征层：128 通道特征图 ----
        # x 形状: (batch, 128, 15, 15)

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


class CNNV12Model(CNNV11Model):
    """CNN v12 轻量双头残差卷积神经网络模型

    双头输出：策略（225 维落子概率） + 价值（[-1,1] 胜率估值）。

    与 CNNV11Model 的唯一区别是权重路径、模型名与网络结构；
    predict / predict_batch / _prepare_input 全部继承父类，
    相对视角编码与 cnn_v11 逐位一致。

    依赖声明：继承自 ai.models.cnn_v11.model.CNNV11Model。
    """

    # 默认模型保存路径
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_v12.pth")

    @property
    def name(self) -> str:
        """模型名称"""
        return "cnn_v12"

    def _create_network(self) -> nn.Module:
        """创建 CNN v12 神经网络实例

        Returns:
            CNNV12Network 实例
        """
        return CNNV12Network()
