"""FC v1 全连接神经网络模型实现

网络结构：输入层(226) -> 隐藏层1-5(各1000, ReLU) -> 输出层(225) -> LogSoftmax

FCv1Model 继承自 NeuralNetworkModel，只需实现：
- _create_network: 返回 FCv1Network 实例
- _prepare_input: 将 226 维向量转换为 (1, 226) 张量
其余通用逻辑（__init__、predict、save、load）由基类提供。
"""

import os
from typing import List
import torch
import torch.nn as nn
from ai.base_model import NeuralNetworkModel


class FCv1Network(nn.Module):
    """FC v1 神经网络结构
    
    5层隐藏层，每层1000个神经元，使用ReLU激活函数。
    输入层(226) -> 隐藏层1-5(各1000, ReLU) -> 输出层(225) -> LogSoftmax
    """
    
    def __init__(self):
        """初始化网络结构"""
        super(FCv1Network, self).__init__()
        
        # 定义网络层
        # 输入层到第一个隐藏层
        self.fc1 = nn.Linear(226, 1000)
        self.relu1 = nn.ReLU()
        
        # 隐藏层2-5
        self.fc2 = nn.Linear(1000, 1000)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(1000, 1000)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(1000, 1000)
        self.relu4 = nn.ReLU()
        
        self.fc5 = nn.Linear(1000, 1000)
        self.relu5 = nn.ReLU()
        
        # 输出层
        self.fc_output = nn.Linear(1000, 225)
        
        # LogSoftmax 激活函数，dim=1 表示按行计算，输出 log-probability
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, 226)
            
        Returns:
            输出张量，形状为 (batch_size, 225)，log-probability（值 <= 0），exp 后总和为 1
        """
        # 第1层
        x = self.fc1(x)
        x = self.relu1(x)
        
        # 第2层
        x = self.fc2(x)
        x = self.relu2(x)
        
        # 第3层
        x = self.fc3(x)
        x = self.relu3(x)
        
        # 第4层
        x = self.fc4(x)
        x = self.relu4(x)
        
        # 第5层
        x = self.fc5(x)
        x = self.relu5(x)
        
        # 输出层
        x = self.fc_output(x)
        
        # LogSoftmax 激活，输出 log-probability 分布
        x = self.log_softmax(x)
        
        return x


class FCv1Model(NeuralNetworkModel):
    """FC v1 模型
    
    全连接神经网络模型，用于预测五子棋落子位置的 reward。
    继承 NeuralNetworkModel，通用逻辑由基类提供。
    
    默认行为：
    - 如果 fc_v1.pth 存在，自动加载该模型参数
    - 如果不存在，创建新的模型
    - 使用 fresh=True 可强制创建新模型
    """
    
    # 默认模型保存路径
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "fc_v1.pth")
    
    @property
    def name(self) -> str:
        """模型名称"""
        return "fc_v1"
    
    def _create_network(self) -> nn.Module:
        """创建 FC v1 神经网络实例
        
        Returns:
            FCv1Network 实例
        """
        return FCv1Network()
    
    def _prepare_input(self, board_vector: List[int], device: torch.device) -> torch.Tensor:
        """将 226 维向量转换为 FC 模型输入张量 (1, 226)
        
        FC 模型直接使用展平的 226 维向量作为输入。
        
        Args:
            board_vector: 226 维向量
            device: 目标设备
            
        Returns:
            形状为 (1, 226) 的张量
        """
        return torch.tensor(board_vector, dtype=torch.float32, device=device).unsqueeze(0)
