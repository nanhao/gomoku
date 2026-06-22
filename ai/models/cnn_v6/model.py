"""CNN v6 残差卷积神经网络模型实现

CNN v6 在 cnn_v5 的网络结构基础上引入两个增强：
- BatchNorm：每层 Conv 后添加 BN，位于 ReLU 之前，稳定各层输入分布
- 残差连接（Residual Connection）：Conv2+Conv3 组成残差块 1，
  Conv4+Conv5 组成残差块 2，shortcut 直接相加（无需 1×1 投影）

网络结构（详见 design/15/main_docs/forward_flow.puml）：
输入(15×15×3) → Conv1+BN+ReLU → [残差块1: Conv2+BN+ReLU+Conv3+BN→+shortcut→ReLU]
→ [残差块2: Conv4+BN+ReLU+Conv5+BN→+shortcut→ReLU]
→ Conv_out(1×1) → Flatten → LogSoftmax

输入编码（同 cnn_v5）：
- 通道1：黑棋位置（1=黑棋，0=其他）
- 通道2：白棋位置（1=白棋，0=其他）
- 通道3：当前落子方（全1=黑方，全0=白方）

输出：LogSoftmax log-probability 分布，值域 (-∞, 0]，exp 后总和为 1

CNNV6Model 继承自 NeuralNetworkModel，只需实现：
- _create_network: 返回 CNNV6Network 实例
- _prepare_input: 调用 board_vector_to_cnn_tensor 转换输入
其余通用逻辑（__init__、predict、save、load）由基类提供。
"""

import os
from typing import List
import torch
import torch.nn as nn
from ai.base_model import NeuralNetworkModel, board_vector_to_cnn_tensor


class CNNV6Network(nn.Module):
    """CNN v6 残差卷积神经网络结构

    在 cnn_v5 的五层纯卷积基础上引入 BatchNorm 和残差连接：
    - 每层 Conv 后添加 BN → ReLU（Conv→BN→ReLU 顺序）
    - Conv2+Conv3 组成第一个残差块，Conv4+Conv5 组成第二个残差块
    - 残差连接将块输入直接加到第二个 BN 的输出上，再经过 ReLU

    由于所有卷积层使用 padding=1、stride=1、通道保持 32，空间尺寸
    始终为 15×15，shortcut 可直接相加，无需 1×1 投影层。

    输入：(batch, 3, 15, 15) 的棋盘状态
    输出：(batch, 225) 的 LogSoftmax log-probability 分布

    参数量：38,241（vs cnn_v5 的 37,921，仅增加 5 个 BN 层的 320 参数）
    """

    def __init__(self):
        """初始化 CNN v6 网络结构

        创建 5 层 3×3 Conv + BN + ReLU 以及 1 层 1×1 输出 Conv。
        残差连接逻辑在 forward 中内联实现，不封装为独立子模块。
        """
        super(CNNV6Network, self).__init__()

        # ---- 第 1 层：Conv(3→32) + BN + ReLU ----
        # 第一层也加 BN，统一各层行为，稳定初始输入分布（仅增加 64 参数）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        # ---- 第 2 层：Conv(32→32) + BN + ReLU（残差块 1 的第一层）----
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        # ---- 第 3 层：Conv(32→32) + BN（残差块 1 的第二层，后接 shortcut + ReLU）----
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        # ---- 第 4 层：Conv(32→32) + BN + ReLU（残差块 2 的第一层）----
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        # ---- 第 5 层：Conv(32→32) + BN（残差块 2 的第二层，后接 shortcut + ReLU）----
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        # ---- 输出层：1×1 Conv(32→1) 替代 FC ----
        # 不加 ReLU，否则初始化不当会导致所有输出为负值，
        # ReLU 将其全部置零 → LogSoftmax 输入全零 → 均匀分布 → 梯度为零
        self.conv_out = nn.Conv2d(32, 1, kernel_size=1)

        # ---- LogSoftmax 激活，输出 log-probability ----
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        严格按照设计文档中的前向流程：
        1. Conv1 → BN1 → ReLU1
        2. 残差块 1：shortcut = x; x = Conv2 → BN2 → ReLU2 → Conv3 → BN3;
           x = ReLU(BN3_output + shortcut)
        3. 残差块 2：shortcut = x; x = Conv4 → BN4 → ReLU4 → Conv5 → BN5;
           x = ReLU(BN5_output + shortcut)
        4. Conv_out(1×1) → Flatten → LogSoftmax

        Args:
            x: 输入张量，形状 (batch_size, 3, 15, 15)

        Returns:
            输出张量，形状 (batch_size, 225)，log-probability（值 ≤ 0），
            exp 后每行总和为 1
        """
        # ---- Conv1 + BN + ReLU ----
        x = self.relu1(self.bn1(self.conv1(x)))

        # ---- 残差块 1（Conv2 + Conv3） ----
        # 保存 shortcut（块输入）
        shortcut = x
        # 主路径：Conv2 → BN2 → ReLU2 → Conv3 → BN3
        x = self.bn3(self.conv3(self.relu2(self.bn2(self.conv2(x)))))
        # shortcut 连接：BN3 输出 + 块输入，再经过 ReLU3
        x = self.relu3(x + shortcut)

        # ---- 残差块 2（Conv4 + Conv5） ----
        # 保存 shortcut（块输入）
        shortcut = x
        # 主路径：Conv4 → BN4 → ReLU4 → Conv5 → BN5
        x = self.bn5(self.conv5(self.relu4(self.bn4(self.conv4(x)))))
        # shortcut 连接：BN5 输出 + 块输入，再经过 ReLU5
        x = self.relu5(x + shortcut)

        # ---- 输出层 ----
        # 1×1 卷积：32 通道 → 1 通道，保持 15×15 空间结构
        x = self.conv_out(x)

        # 展平：(batch, 1, 15, 15) → (batch, 225)
        x = x.view(x.size(0), -1)

        # LogSoftmax 激活
        x = self.log_softmax(x)

        return x


class CNNV6Model(NeuralNetworkModel):
    """CNN v6 残差卷积神经网络模型

    在 cnn_v5 基础上引入 BatchNorm 和残差连接的卷积神经网络模型，
    用于预测五子棋落子位置的概率。
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
        """模型名称

        Returns:
            "cnn_v6"
        """
        return "cnn_v6"

    def _create_network(self) -> nn.Module:
        """创建 CNN v6 神经网络实例

        Returns:
            CNNV6Network 实例，包含 BN 层和残差连接
        """
        return CNNV6Network()

    def _prepare_input(self, board_vector: List[int],
                       device: torch.device) -> torch.Tensor:
        """将 226 维向量转换为 CNN 输入张量 (1, 3, 15, 15)

        使用 board_vector_to_cnn_tensor 工具函数完成转换，
        该函数是所有 CNN 模型共用的输入转换逻辑。

        Args:
            board_vector: 226 维向量（225 棋盘 + 1 当前方）
            device: 目标设备（CPU 或 GPU）

        Returns:
            形状为 (1, 3, 15, 15) 的张量
        """
        return board_vector_to_cnn_tensor(board_vector, device)
