"""AI 模型基类模块

定义所有 AI 模型必须实现的接口。
所有模型输出统一为 Softmax 概率分布。
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseModel(ABC):
    """AI 模型基类
    
    所有 AI 模型必须继承此类并实现以下接口：
    - predict: 预测每个落子位置的概率
    - name: 模型名称
    - save: 保存模型参数到文件
    - load: 从文件加载模型参数
    - network: 神经网络模块（RandomModel 使用 None）
    
    注意：network 是实例属性而非抽象方法，因为子类在 __init__ 中赋值。
    BaseTrainer 依赖 self.model.network 存在，因此所有子类必须设置此属性。
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
    
    @abstractmethod
    def save(self, model_path: str = None) -> str:
        """保存模型参数到文件
        
        Args:
            model_path: 保存路径，为 None 则使用默认路径
            
        Returns:
            模型保存的文件路径
        """
        pass
    
    @abstractmethod
    def load(self, model_path: str = None):
        """从文件加载模型参数
        
        Args:
            model_path: 模型文件路径，为 None 则使用默认路径
        """
        pass
    
    def _ensure_network_attribute(self):
        """确保 network 属性已设置（供 BaseTrainer 初始化时调用）
        
        如果子类未设置 network 属性，抛出 AttributeError。
        """
        if not hasattr(self, 'network') or self.network is None:
            if not hasattr(self, '_network_impl'):
                raise AttributeError(
                    f"{self.__class__.__name__} 必须设置 self.network 属性。"
                    f"对于使用神经网络的模型，在 __init__ 中设置 self.network = <Network>；"
                    f"对于 RandomModel，设置 self.network = None。"
                )
