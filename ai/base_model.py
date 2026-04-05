"""AI 模型基类模块

定义所有 AI 模型必须实现的接口。
所有模型输出统一为 Softmax 概率分布。
"""

from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    """AI 模型基类
    
    所有 AI 模型必须继承此类并实现 predict 方法。
    """
    
    @abstractmethod
    def predict(self, board_vector: List[int]) -> List[float]:
        """
        预测每个落子位置的概率
        
        Args:
            board_vector: 226 维向量，前 225 位是棋盘（按行展开，0=空, 1=黑, 2=白），
                         最后 1 位是当前落子方（1=黑, 2=白）
            
        Returns:
            225 维列表，格式为 [prob(0,0), prob(0,1), ..., prob(14,14)]
            元素索引 i 对应棋盘位置 (i//15, i%15)，概率范围：0 <= prob <= 1
            所有概率之和为 1（Softmax 概率分布）
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """模型名称"""
        pass
