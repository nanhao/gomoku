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

    AI 名称到模型的映射由 ai.registry 统一维护，外部调用者无需了解具体模型类。

    与 get_trainer 的行为差异：
    - create_engine 面向用户交互场景，未知名称时回退到 RandomModel（容错），
      保证 GUI 不因配置错误而崩溃，同时输出 warning 日志。
    - get_trainer 面向开发者训练场景，未知名称时抛出 ValueError（快速失败），
      帮助开发者尽早发现配置问题。

    Args:
        ai_name: AI 名称，如 "random"、"fc_v1"、"cnn_v1"

    Returns:
        对应的 AIEngine 实例；未知名称时回退到 AIEngine(RandomModel) 并输出 warning
    """
    from ai.registry import get_model, list_models

    try:
        model = get_model(ai_name)
    except ValueError:
        import logging
        logging.getLogger(__name__).warning(
            f"未知 AI 名称 '{ai_name}'，回退到 RandomModel。"
            f"可选值: {list_models()}"
        )
        model = get_model("random")
    return AIEngine(model)


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
    
    def get_move(self, board: List[List[int]], current_player: int, probs: List[float] = None) -> Tuple[int, int]:
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
            probs: 预计算的概率分布（225维），若提供则跳过 model.predict 调用
            
        Returns:
            (x, y) 落子位置，0 <= x < 15, 0 <= y < 15
            
        Raises:
            ValueError: 棋盘已满，无空位可落子
        """
        # 1. 转换棋盘为向量
        board_vector = self.board_to_vector(board, current_player)
        
        # 2. 获取模型预测（225 维概率分布）
        # 性能优化：支持传入预计算的 probs，避免重复调用 model.predict
        if probs is None:
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
        # 如果所有有效概率都为 0（模型可能选择了已占位置），回退到均匀分布
        total_prob = sum(valid_probs)
        if total_prob <= 0:
            valid_probs = [1.0] * len(valid_positions)
        
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
