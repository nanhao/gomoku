"""Random AI 模型实现

第一版 AI，对空位返回均匀概率分布，使引擎随机选择空位落子。
"""

from typing import List
from ai.base_model import BaseModel


class RandomModel(BaseModel):
    """随机 AI 模型
    
    对空位返回均匀概率分布，已落子位置概率为 0。
    引擎根据概率随机选择落子位置。
    """
    
    @property
    def name(self) -> str:
        """模型名称"""
        return "random"
    
    def predict(self, board_vector: List[int]) -> List[float]:
        """
        预测每个落子位置的概率
        
        对空位（board 上为 0 的位置）返回均匀概率分布，
        已落子位置（board 上为 1 或 2）概率为 0。
        所有空位的概率总和为 1。
        
        Args:
            board_vector: 226 维向量，前 225 位是棋盘状态
            
        Returns:
            225 维概率列表，索引 i 对应位置 (i//15, i%15)
        """
        # 获取棋盘状态（前 225 位）
        board = board_vector[:225]
        
        # 统计空位数量
        empty_count = sum(1 for cell in board if cell == 0)
        
        if empty_count == 0:
            # 棋盘已满，返回均匀分布以遵守接口契约（概率之和为1）
            return [1.0 / 225] * 225
        
        # 空位均匀分布概率
        prob = 1.0 / empty_count
        
        # 构建概率列表：空位为 prob，已落子位置为 0
        return [prob if cell == 0 else 0.0 for cell in board]
