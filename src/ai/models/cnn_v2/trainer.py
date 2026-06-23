"""CNN v2 训练器实现

负责训练 CNN v2 轻量化模型。
"""

from typing import Optional, List
import torch
from ai.base_trainer import BaseTrainer
from ai.base_model import board_vector_to_cnn_tensor
from .model import CNNV2Model


class CNNV2Trainer(BaseTrainer):
    """CNN v2 训练器

    负责训练 CNN v2 模型，提供单轮次训练和批量训练。
    继承自 BaseTrainer，使用通用训练逻辑。

    默认行为：
    - 尝试加载已保存的模型 cnn_v2.pth
    - 如果不存在，创建新模型
    - 使用 fresh=True 可强制创建新模型
    """

    def _create_model(self, fresh: bool) -> CNNV2Model:
        """创建 CNN v2 模型实例

        Args:
            fresh: 是否创建新模型（不加载已保存的模型）

        Returns:
            CNNV2Model 实例
        """
        return CNNV2Model(fresh=fresh)

    @property
    def model_name(self) -> str:
        """模型名称

        Returns:
            "cnn_v2"
        """
        return "cnn_v2"

    def _convert_inputs(self, inputs: List[List[int]]) -> torch.Tensor:
        """将输入数据转换为 CNN 所需的 4D 张量格式

        CNN 模型期望输入形状为 (N, 3, 15, 15)，其中：
        - N: batch size
        - 3: 通道数（黑棋、白棋、当前方）
        - 15x15: 棋盘大小

        使用 board_vector_to_cnn_tensor 工具函数完成单个向量的转换，
        然后将 batch 维度合并。

        Args:
            inputs: 输入数据列表，每个元素是 226 维向量

        Returns:
            输入张量，形状为 (N, 3, 15, 15)
        """
        batch_tensors = []

        for board_vector in inputs:
            # 使用共用的工具函数转换，先在 CPU 上创建张量
            # 后续由 BaseTrainer._prepare_tensors 统一 .to(device) 移到 GPU
            tensor = board_vector_to_cnn_tensor(board_vector, torch.device("cpu"))
            batch_tensors.append(tensor)

        # 合并 batch: [(1, 3, 15, 15), ...] -> (N, 3, 15, 15)
        return torch.cat(batch_tensors, dim=0)
