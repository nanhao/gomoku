"""CNN 卷积神经网络模型实现

网络结构：
输入层(15x15x3) -> 卷积层 -> 全连接层 -> 输出层(225) -> Softmax

输入编码：
- 通道1：黑棋位置（1表示黑棋，0表示其他）
- 通道2：白棋位置（1表示白棋，0表示其他）
- 通道3：当前落子方（全1表示黑方，全0表示白方）

输出：Softmax 概率分布，值域 (0, 1)，总和为 1

CNNV1Model 继承自 NeuralNetworkModel，只需实现：
- _create_network: 返回 CNNV1Network 实例
- _prepare_input: 调用 board_vector_to_cnn_tensor 转换输入
其余通用逻辑（__init__、predict、save、load）由基类提供。
"""

import os
from typing import List
import torch
import torch.nn as nn
from ai.base_model import NeuralNetworkModel, board_vector_to_cnn_tensor


class CNNV1Network(nn.Module):
    """CNN 神经网络结构
    
    轻量化卷积神经网络用于五子棋棋盘特征提取和落子位置预测。
    输入：15x15x3 的棋盘状态
    输出：225 维的 Softmax 概率分布
    
    网络结构：
    - 3 层卷积：3→32→64→128
    - 2 层全连接：28800→150→225
    - 参数量：约 445 万（与 FC v1 相当）
    """
    
    def __init__(self):
        """初始化网络结构"""
        super(CNNV1Network, self).__init__()
        
        # 卷积层部分（轻量化设计）
        # 输入: 3通道 (黑棋、白棋、当前方), 15x15
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        
        # 全连接层部分
        # 128通道 * 15 * 15 = 28800
        self.fc1 = nn.Linear(128 * 15 * 15, 150)
        self.relu_fc1 = nn.ReLU()
        
        # 输出层: 225个位置
        self.fc_output = nn.Linear(150, 225)
        
        # Softmax 激活函数
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, 3, 15, 15)
            
        Returns:
            输出张量，形状为 (batch_size, 225)，值域 (0, 1)，每行和为 1
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
        
        # Softmax 激活，输出概率分布
        x = self.softmax(x)
        
        return x


class CNNV1Model(NeuralNetworkModel):
    """CNN 模型
    
    卷积神经网络模型，用于预测五子棋落子位置的概率。
    继承 NeuralNetworkModel，通用逻辑由基类提供。
    
    默认行为：
    - 如果 cnn.pth 存在，自动加载该模型参数
    - 如果不存在，创建新的模型
    - 使用 fresh=True 可强制创建新模型
    """
    
    # 默认模型保存路径
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn.pth")
    
    @property
    def name(self) -> str:
        """模型名称"""
        return "cnn_v1"
    
    def _create_network(self) -> nn.Module:
        """创建 CNN v1 神经网络实例
        
        Returns:
            CNNV1Network 实例
        """
        return CNNV1Network()
    
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
