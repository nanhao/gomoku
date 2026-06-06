"""CNN v2 卷积神经网络模型实现

CNN v1 的轻量化改进版本。

改进点：
- 卷积层统一为 64 通道：3→64→64→64
- FC1 层从 28800→64 进一步压缩为 14400→64
- 总参数量从约 445 万降至约 101 万（-77%）
- 计算量从 23M 降至 18M MACs（-22%）

网络结构：
输入层(15x15x3) -> 卷积层 -> 全连接层 -> 输出层(225) -> LogSoftmax

输入编码：
- 通道1：黑棋位置（1表示黑棋，0表示其他）
- 通道2：白棋位置（1表示白棋，0表示其他）
- 通道3：当前落子方（全1表示黑方，全0表示白方）

输出：LogSoftmax log-probability 分布，值域 (-inf, 0]，exp 后总和为 1

CNNV2Model 继承自 NeuralNetworkModel，只需实现：
- _create_network: 返回 CNNV2Network 实例
- _prepare_input: 调用 board_vector_to_cnn_tensor 转换输入
其余通用逻辑（__init__、predict、save、load）由基类提供。
"""

import os
from typing import List
import torch
import torch.nn as nn
from ai.base_model import NeuralNetworkModel, board_vector_to_cnn_tensor


class CNNV2Network(nn.Module):
    """CNN v2 神经网络结构

    轻量化卷积神经网络用于五子棋棋盘特征提取和落子位置预测。
    相对 CNN v1，卷积层统一为 64 通道，FC1 输入从 28800 降至 14400。

    输入：15x15x3 的棋盘状态
    输出：225 维的 LogSoftmax log-probability 分布

    网络结构：
    - 3 层卷积：3→64→64→64（统一通道数，减少参数量）
    - 2 层全连接：14400→64→225
    - 参数量：约 101 万（CNN v1 的 23%）
    """

    def __init__(self):
        """初始化网络结构"""
        super(CNNV2Network, self).__init__()

        # 卷积层部分（统一 64 通道，进一步轻量化）
        # 输入: 3通道 (黑棋、白棋、当前方), 15x15
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        # 全连接层部分（轻量化改进）
        # 64通道 * 15 * 15 = 14400
        # CNN v2 改进：统一卷积通道后 FC1 输入从 28800 降至 14400
        self.fc1 = nn.Linear(64 * 15 * 15, 64)
        self.relu_fc1 = nn.ReLU()

        # 输出层: 225个位置
        self.fc_output = nn.Linear(64, 225)

        # LogSoftmax 激活函数，输出 log-probability
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量，形状为 (batch_size, 3, 15, 15)

        Returns:
            输出张量，形状为 (batch_size, 225)，log-probability（值 <= 0），exp 后总和为 1
        """
        # 卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = self.relu_fc1(x)

        # 输出层
        x = self.fc_output(x)

        # LogSoftmax 激活，输出 log-probability 分布
        x = self.log_softmax(x)

        return x


class CNNV2Model(NeuralNetworkModel):
    """CNN v2 模型

    卷积神经网络模型，用于预测五子棋落子位置的概率。
    CNN v1 的轻量化版本，参数量减少 77%。
    继承 NeuralNetworkModel，通用逻辑由基类提供。

    默认行为：
    - 如果 cnn_v2.pth 存在，自动加载该模型参数
    - 如果不存在，创建新的模型
    - 使用 fresh=True 可强制创建新模型
    """

    # 默认模型保存路径
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_v2.pth")

    @property
    def name(self) -> str:
        """模型名称"""
        return "cnn_v2"

    def _create_network(self) -> nn.Module:
        """创建 CNN v2 神经网络实例

        Returns:
            CNNV2Network 实例
        """
        return CNNV2Network()

    def _prepare_input(self, board_vector: List[int], device: torch.device) -> torch.Tensor:
        """将 226 维向量转换为 CNN 输入张量 (1, 3, 15, 15)

        使用 board_vector_to_cnn_tensor 工具函数完成转换，
        该函数是 CNNV1Model 和 CNNV2Model 共用的输入转换逻辑。

        Args:
            board_vector: 226 维向量
            device: 目标设备

        Returns:
            形状为 (1, 3, 15, 15) 的张量
        """
        return board_vector_to_cnn_tensor(board_vector, device)
