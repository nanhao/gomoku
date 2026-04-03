"""Random AI 模型实现

第一版 AI，所有位置返回 reward=0，使引擎随机选择空位落子。
"""

from typing import List
from ai.base_model import BaseModel


class RandomModel(BaseModel):
    """随机 AI 模型
    
    无论输入的棋盘状态如何，返回所有位置 reward=0。
    由于所有位置 reward 相同，引擎会随机选择空位落子。
    """
    
    @property
    def name(self) -> str:
        """模型名称"""
        return "random"
    
    def predict(self, board_vector: List[int]) -> List[int]:
        """
        预测每个落子位置的 reward
        
        无论输入如何，所有 225 个位置都返回 reward=0。
        
        Args:
            board_vector: 226 维向量（未使用）
            
        Returns:
            225 维列表，所有 reward 为 0。索引 i 对应位置 (i//15, i%15)
        """
        return [0] * 225
