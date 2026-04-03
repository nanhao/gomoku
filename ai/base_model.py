"""AI 模型基类模块

定义所有 AI 模型必须实现的接口。
"""

from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    """AI 模型基类
    
    所有 AI 模型必须继承此类并实现 predict 方法。
    """
    
    @abstractmethod
    def predict(self, board_vector: List[int]) -> List[int]:
        """
        预测每个落子位置的 reward
        
        Args:
            board_vector: 226 维向量，前 225 位是棋盘（按行展开，0=空, 1=黑, 2=白），
                         最后 1 位是当前落子方（1=黑, 2=白）
            
        Returns:
            225 维列表，格式为 [reward(0,0), reward(0,1), ..., reward(14,14)]
            元素索引 i 对应棋盘位置 (i//15, i%15)，reward 范围：-1000000 <= reward <= 1000000
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """模型名称"""
        pass
