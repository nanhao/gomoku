"""CNN 卷积神经网络模型实现

网络结构：
输入层(15x15x3) -> 卷积层 -> 全连接层 -> 输出层(225) -> Softmax

输入编码：
- 通道1：黑棋位置（1表示黑棋，0表示其他）
- 通道2：白棋位置（1表示白棋，0表示其他）
- 通道3：当前落子方（全1表示黑方，全0表示白方）

输出：Softmax 概率分布，值域 (0, 1)，总和为 1
"""

import os
from typing import List
import torch
import torch.nn as nn
from ai.base_model import BaseModel


class CNNNetwork(nn.Module):
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
        super(CNNNetwork, self).__init__()
        
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


class CNNModel(BaseModel):
    """CNN 模型
    
    卷积神经网络模型，用于预测五子棋落子位置的概率。
    
    默认行为：
    - 如果 cnn.pth 存在，自动加载该模型参数
    - 如果不存在，创建新的模型
    - 使用 fresh=True 可强制创建新模型
    """
    
    # 默认模型保存路径
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn.pth")
    
    def __init__(self, model_path: str = None, fresh: bool = False):
        """初始化模型
        
        Args:
            model_path: 模型文件路径，如果存在则加载
            fresh: 是否强制创建新模型（不加载已保存的模型）
        """
        super(CNNModel, self).__init__()
        
        # 创建神经网络
        self.network = CNNNetwork()
        
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
        return "cnn_v1"
    
    def _vector_to_tensor(self, board_vector: List[int], device: torch.device = None) -> torch.Tensor:
        """将 226 维向量转换为 CNN 输入张量 (1, 3, 15, 15)
        
        Args:
            board_vector: 226 维向量，前 225 位是棋盘，最后 1 位是当前落子方
            device: 目标设备，如果为 None 则使用 CPU
            
        Returns:
            形状为 (1, 3, 15, 15) 的张量
        """
        if device is None:
            device = torch.device("cpu")
            
        # 提取棋盘和当前落子方
        board = board_vector[:225]
        current_player = board_vector[225] if len(board_vector) > 225 else 1
        
        # 创建 3 个通道
        # 通道1: 黑棋位置
        black_channel = [1.0 if x == 1 else 0.0 for x in board]
        # 通道2: 白棋位置
        white_channel = [1.0 if x == 2 else 0.0 for x in board]
        # 通道3: 当前落子方（全1表示黑方，全0表示白方）
        player_channel = [1.0 if current_player == 1 else 0.0] * 225
        
        # 重塑为 15x15
        black_matrix = [black_channel[i*15:(i+1)*15] for i in range(15)]
        white_matrix = [white_channel[i*15:(i+1)*15] for i in range(15)]
        player_matrix = [player_channel[i*15:(i+1)*15] for i in range(15)]
        
        # 转换为张量并堆叠
        black_tensor = torch.tensor(black_matrix, dtype=torch.float32, device=device)
        white_tensor = torch.tensor(white_matrix, dtype=torch.float32, device=device)
        player_tensor = torch.tensor(player_matrix, dtype=torch.float32, device=device)
        
        # 堆叠为 (3, 15, 15) 并添加 batch 维度
        input_tensor = torch.stack([black_tensor, white_tensor, player_tensor], dim=0)
        input_tensor = input_tensor.unsqueeze(0)  # (1, 3, 15, 15)
        
        return input_tensor
    
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
            
        # 将输入转换为 CNN 输入张量并移到对应设备
        input_tensor = self._vector_to_tensor(board_vector, device=device)
            
        # 前向传播
        self.network.eval()
        with torch.no_grad():
            output_tensor = self.network(input_tensor)
    
        # 将输出转换为列表 (值域 (0, 1)，Softmax 概率)
        output_values = output_tensor.squeeze(0).tolist()
    
        # 构建 225 维结果，索引 i 对应位置 (i//15, i%15)
        # 直接返回概率值
        result = [float(prob) for prob in output_values]
    
        return result
    
    def save(self, model_path: str = None) -> str:
        """保存模型参数
        
        Args:
            model_path: 保存路径，默认使用 ai/models/cnn/cnn.pth
            
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
