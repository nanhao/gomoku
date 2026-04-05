"""FC v1 全连接神经网络模型实现

网络结构：输入层(226) -> 隐藏层1-5(各1000, ReLU) -> 输出层(225) -> Softmax
"""

import os
from typing import List
import torch
import torch.nn as nn
from ai.base_model import BaseModel


class FCv1Network(nn.Module):
    """FC v1 神经网络结构
    
    5层隐藏层，每层1000个神经元，使用ReLU激活函数。
    输入层(226) -> 隐藏层1-5(各1000, ReLU) -> 输出层(225) -> Softmax
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
        
        # Softmax 激活函数，dim=1 表示按行计算
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, 226)
            
        Returns:
            输出张量，形状为 (batch_size, 225)，值域 (0, 1)，每行和为 1
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
        
        # Softmax 激活，输出概率分布
        x = self.softmax(x)
        
        return x


class FCv1Model(BaseModel):
    """FC v1 模型
    
    全连接神经网络模型，用于预测五子棋落子位置的 reward。
    
    默认行为：
    - 如果 fc_v1.pth 存在，自动加载该模型参数
    - 如果不存在，创建新的模型
    - 使用 fresh=True 可强制创建新模型
    """
    
    # 默认模型保存路径
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "fc_v1.pth")
    
    def __init__(self, model_path: str = None, fresh: bool = False):
        """初始化模型
        
        Args:
            model_path: 模型文件路径，如果存在则加载
            fresh: 是否强制创建新模型（不加载已保存的模型）
        """
        super(FCv1Model, self).__init__()

        # 创建神经网络
        self.network = FCv1Network()
        
        # 自动检测并使用 GPU
        if torch.cuda.is_available():
            self.network.to('cuda')

        # fresh=True 时，不加载任何模型，使用随机初始化的参数
        if fresh:
            return  # 使用随机初始化的参数

        # 如果指定了 model_path 且文件存在，加载该模型
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        # 否则尝试加载默认路径的模型
        elif os.path.exists(self.DEFAULT_MODEL_PATH):
            self.load(self.DEFAULT_MODEL_PATH)
    
    @property
    def name(self) -> str:
        """模型名称"""
        return "fc_v1"
    
    def predict(self, board_vector: List[int]) -> List[float]:
        """预测每个落子位置的概率
            
        Args:
            board_vector: 226 维向量，前 225 位是棋盘（按行展开，0=空, 1=黑, 2=白），
                         最后 1 位是当前落子方（1=黑, 2=白）
                
        Returns:
            225 维列表，格式为 [prob(0,0), prob(0,1), ..., prob(14,14)]
            索引 i 对应位置 (i//15, i%15)，概率范围：0 < prob < 1
            所有概率之和为 1（Softmax 输出）
        """
        # 获取设备（与网络所在设备一致）
        device = next(self.network.parameters()).device
        
        # 将输入转换为张量并移到对应设备
        input_tensor = torch.tensor(board_vector, dtype=torch.float32, device=device).unsqueeze(0)
            
        # 前向传播
        self.network.eval()
        with torch.no_grad():
            output_tensor = self.network(input_tensor)
            
        # 将输出转换为列表 (值域 (0, 1)，Softmax 概率)
        output_values = output_tensor.squeeze(0).tolist()
            
        # 构建 225 维结果，索引 i 对应位置 (i//15, i%15)
        # 直接返回概率值，不再映射到 [-1000000, 1000000]
        result = [float(prob) for prob in output_values]
            
        return result
    
    def save(self, model_path: str = None) -> str:
        """保存模型参数
        
        Args:
            model_path: 保存路径，默认使用 ai/fc_v1/fc_v1.pth
            
        Returns:
            实际保存的路径
        """
        if model_path is None:
            model_path = self.DEFAULT_MODEL_PATH
        
        # 保存模型状态字典
        torch.save(self.network.state_dict(), model_path)
        
        return model_path
    
    def load(self, model_path: str):
        """加载模型参数
        
        Args:
            model_path: 模型文件路径
        """
        self.network.load_state_dict(torch.load(model_path, weights_only=True))
        self.network.eval()
