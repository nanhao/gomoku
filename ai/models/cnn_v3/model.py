"""CNN v3 极简卷积神经网络模型实现

CNN v3 追求结构最简化：
- 三层卷积统一 32 通道，无 BatchNorm，无 Dropout
- FC 层直接从卷积输出映射到 225 维，无中间隐藏层
- Conv → ReLU 直接堆叠，FC → Softmax 直接输出

网络结构：
输入层(15x15x3) -> 卷积层(3层32通道) -> FC(7200->225) -> Softmax

输入编码：
- 通道1：黑棋位置（1表示黑棋，0表示其他）
- 通道2：白棋位置（1表示白棋，0表示其他）
- 通道3：当前落子方（全1表示黑方，全0表示白方）

输出：Softmax 概率分布，值域 (0, 1)，总和为 1

CNNV3Model 继承自 NeuralNetworkModel，只需实现：
- _create_network: 返回 CNNV3Network 实例
- _prepare_input: 调用 board_vector_to_cnn_tensor 转换输入
其余通用逻辑（__init__、predict、save、load）由基类提供。
"""

import os
from typing import List
import torch
import torch.nn as nn
from ai.base_model import NeuralNetworkModel, board_vector_to_cnn_tensor


class CNNV3Network(nn.Module):
    """CNN v3 极简神经网络结构

    三层 32 通道卷积（无 BN、无 Dropout），FC 直接输出 225 维。
    结构最简化，探索极简网络在五子棋任务上的学习能力。

    输入：15x15x3 的棋盘状态
    输出：225 维的 Softmax 概率分布

    网络结构：
    - 3 层卷积：3→32→32→32（统一 32 通道，无 BN）
    - 1 层全连接：7200→225（直接映射，无隐藏层）
    - Softmax 输出
    - 参数量：约 164 万
    """

    def __init__(self):
        """初始化网络结构"""
        super(CNNV3Network, self).__init__()

        # 卷积层部分（统一 32 通道，无 BN，无 Dropout）
        # 输入: 3通道 (黑棋、白棋、当前方), 15x15
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        # 全连接层：直接从卷积输出映射到 225 维
        # 32通道 * 15 * 15 = 7200
        self.fc = nn.Linear(32 * 15 * 15, 225)

        # Softmax 激活函数
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量，形状为 (batch_size, 3, 15, 15)

        Returns:
            输出张量，形状为 (batch_size, 225)，值域 (0, 1)，每行和为 1
        """
        # 卷积层（Conv → ReLU，无 BN）
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))

        # 展平: (batch, 32, 15, 15) -> (batch, 7200)
        x = x.view(x.size(0), -1)

        # FC 直接输出 + Softmax
        x = self.softmax(self.fc(x))

        return x


class CNNV3Model(NeuralNetworkModel):
    """CNN v3 极简模型

    极简卷积神经网络模型，用于预测五子棋落子位置的概率。
    三层 32 通道卷积 + FC 直出，无 BN、无 Dropout。
    继承 NeuralNetworkModel，通用逻辑由基类提供。

    默认行为：
    - 如果 cnn_v3.pth 存在，自动加载该模型参数
    - 如果不存在，创建新的模型
    - 使用 fresh=True 可强制创建新模型
    """

    # 默认模型保存路径
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_v3.pth")

    @property
    def name(self) -> str:
        """模型名称"""
        return "cnn_v3"

    def _create_network(self) -> nn.Module:
        """创建 CNN v3 神经网络实例

        Returns:
            CNNV3Network 实例
        """
        return CNNV3Network()

    def _prepare_input(self, board_vector: List[int], device: torch.device) -> torch.Tensor:
        """将 226 维向量转换为 CNN 输入张量 (1, 3, 15, 15)

        使用 board_vector_to_cnn_tensor 工具函数完成转换，
        该函数是所有 CNN 模型共用的输入转换逻辑。

        Args:
            board_vector: 226 维向量
            device: 目标设备

        Returns:
            形状为 (1, 3, 15, 15) 的张量
        """
        return board_vector_to_cnn_tensor(board_vector, device)
