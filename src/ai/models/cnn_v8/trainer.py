"""CNN v8 训练器实现

负责训练 CNN v8 模型（MCTS 新体系）。
继承 DualHeadTrainer，使用双头损失（MSE + CrossEntropy）。
使用相对视角编码。
"""

from typing import List, Optional

import torch

from ai.dual_head_trainer import DualHeadTrainer
from ai.base_model import board_vector_to_cnn_tensor_relative
from .model import CNNV8Model


class CNNV8Trainer(DualHeadTrainer):
    """CNN v8 训练器（MCTS 新体系）

    继承 DualHeadTrainer，使用双头训练（策略+价值）。
    覆盖 _convert_inputs 使用相对视角编码。

    Attributes:
        model: CNNV8Model 实例
        optimizer: AdamW 优化器
        device: 训练设备
    """

    def _create_model(self, fresh: bool) -> CNNV8Model:
        """创建 CNN v8 模型实例

        Args:
            fresh: 是否创建新模型（不加载已保存的模型）

        Returns:
            CNNV8Model 实例
        """
        gen = getattr(self, 'gen', None)
        return CNNV8Model(fresh=fresh, gen=gen) if gen is not None else CNNV8Model(fresh=fresh)

    @property
    def model_name(self) -> str:
        """模型名称

        Returns:
            "cnn_v8"
        """
        return "cnn_v8"

    def _convert_inputs(self, inputs: List[List[int]]) -> torch.Tensor:
        """将输入数据转换为相对视角 CNN 所需的 4D 张量格式

        CNN 模型期望输入形状为 (N, 3, 15, 15)。
        使用相对视角编码（AlphaGo Zero 风格）。

        Args:
            inputs: 输入数据列表，每个元素是 226 维 board_vector

        Returns:
            输入张量，形状为 (N, 3, 15, 15)
        """
        batch_tensors = []
        for board_vector in inputs:
            tensor = board_vector_to_cnn_tensor_relative(board_vector, torch.device("cpu"))
            batch_tensors.append(tensor)

        return torch.cat(batch_tensors, dim=0)
