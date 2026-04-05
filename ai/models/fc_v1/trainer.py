"""FC v1 训练器实现

负责训练 FC v1 模型。
"""

from typing import Optional
from ai.base_trainer import BaseTrainer
from .model import FCv1Model


class FCv1Trainer(BaseTrainer):
    """FC v1 训练器
    
    负责训练 FC v1 模型，支持单轮次训练和训练周期模式。
    继承自 BaseTrainer，使用通用训练逻辑。
    
    默认行为：
    - 尝试加载已保存的模型 fc_v1.pth
    - 如果不存在，创建新模型
    - 使用 fresh=True 可强制创建新模型
    """
    
    def _create_model(self, fresh: bool) -> FCv1Model:
        """创建 FC v1 模型实例
        
        Args:
            fresh: 是否创建新模型（不加载已保存的模型）
            
        Returns:
            FCv1Model 实例
        """
        return FCv1Model(fresh=fresh)
    
    @property
    def model_name(self) -> str:
        """模型名称
        
        Returns:
            "fc_v1"
        """
        return "fc_v1"
    

