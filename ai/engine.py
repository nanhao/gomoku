"""AI 引擎模块

连接 UI 和 AI 模型，处理输入输出，选择落子位置。
"""

import random
import math
from typing import List, Tuple
from ai.base_model import BaseModel


def create_engine(ai_name: str) -> "AIEngine":
    """
    根据 AI 名称创建对应的 AIEngine 实例。

    AI 名称到模型的映射由 ai 包统一维护，外部调用者无需了解具体模型类。

    Args:
        ai_name: AI 名称，如 "random"、"fc_v1"、"cnn"

    Returns:
        对应的 AIEngine 实例；未知名称时返回 AIEngine(RandomModel)（兜底）
    """
    from ai.models.random import RandomModel
    from ai.models.fc_v1 import FCv1Model
    from ai.models.cnn_v1 import CNNModel

    model_map = {
        "random": RandomModel,
        "fc_v1": FCv1Model,
        "cnn": CNNModel,
    }
    model_cls = model_map.get(ai_name, RandomModel)
    return AIEngine(model_cls())


class AIEngine:
    """AI 引擎 - 连接 UI 和 AI 模型
    
    负责：
    1. 将棋盘状态转换为模型输入向量
    2. 调用模型获取 reward 预测
    3. 根据 reward 选择最佳落子位置（排除已有棋子的位置）
    4. 多个位置 reward 相同时随机选择
    """
    
    def __init__(self, model: BaseModel):
        """
        初始化 AI 引擎
        
        Args:
            model: AI 模型实例，必须继承 BaseModel
        """
        self.model = model
    
    def get_move(self, board: List[List[int]], current_player: int) -> Tuple[int, int]:
        """
        获取 AI 落子位置
        
        流程：
        1. 将棋盘转换为 226 维向量
        2. 调用模型获取 225 维概率分布
        3. 按概率采样选择落子位置（排除已有棋子）
        
        使用概率采样而非选择最大值，确保即使模型输出接近均匀分布，
        也会有足够的随机性，避免自我对弈时出现确定性结果。
        
        Args:
            board: 15x15 棋盘，0=空, 1=黑, 2=白
            current_player: 当前落子方，1=黑, 2=白
            
        Returns:
            (x, y) 落子位置，0 <= x < 15, 0 <= y < 15
            
        Raises:
            ValueError: 棋盘已满，无空位可落子
        """
        # 1. 转换棋盘为向量
        board_vector = self.board_to_vector(board, current_player)
        
        # 2. 获取模型预测（225 维概率分布）
        probs = self.model.predict(board_vector)
        
        # 3. 构建位置-概率列表，排除已有棋子的位置
        valid_positions = []
        valid_probs = []
        for i in range(225):
            x = i // 15
            y = i % 15
            if 0 <= x < 15 and 0 <= y < 15 and board[x][y] == 0:
                valid_positions.append((x, y))
                valid_probs.append(probs[i])
        
        if not valid_positions:
            raise ValueError("棋盘已满，无空位可落子")
        
        # 4. 按概率采样选择位置
        # 使用 random.choices 进行加权随机选择
        chosen_idx = random.choices(range(len(valid_positions)), weights=valid_probs, k=1)[0]
        return valid_positions[chosen_idx]
    
    @staticmethod
    def board_to_vector(board: List[List[int]], current_player: int) -> List[int]:
        """
        将棋盘转换为 226 维向量
        
        转换规则：
        - 前 225 位：按行优先展开棋盘，board[0][0], board[0][1], ..., board[14][14]
        - 第 226 位：当前落子方（1=黑, 2=白）
        
        Args:
            board: 15x15 棋盘
            current_player: 当前落子方
            
        Returns:
            226 维向量
        """
        vector = []
        for row in board:
            vector.extend(row)
        vector.append(current_player)
        return vector
