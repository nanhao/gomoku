"""样本数据加载与预处理模块

从 base_trainer.py 提取，统一负责从 sample.json 到 PyTorch 张量的全链路转换。
"""

import json
import os
from typing import List, Tuple, Callable, Optional

import torch


class SampleLoader:
    """样本数据加载器"""

    def __init__(self, device: torch.device, input_converter: Optional[Callable] = None):
        """初始化

        Args:
            device: PyTorch 设备（CPU 或 GPU）
            input_converter: 输入转换函数，默认直接转为 float32 张量
        """
        self.device = device
        self.input_converter = input_converter

    def load(self, sample_path: str) -> list:
        """加载单个 sample.json 数据

        Args:
            sample_path: sample.json 文件路径

        Returns:
            sample 数据列表

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON 格式错误
        """
        with open(sample_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_multiple(self, sample_dirs: List[str]) -> list:
        """加载多个 sample.json 文件并合并

        Args:
            sample_dirs: 包含 sample.json 的目录路径列表

        Returns:
            合并后的 sample 数据列表

        Raises:
            FileNotFoundError: 某个目录下不存在 sample.json
        """
        all_sample_data = []
        for sample_dir in sample_dirs:
            sample_path = os.path.join(sample_dir, "sample.json")
            if not os.path.exists(sample_path):
                raise FileNotFoundError(f"Sample file not found: {sample_path}")
            all_sample_data.extend(self.load(sample_path))
        return all_sample_data

    def prepare_tensors(self, sample_data: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将 sample 数据转换为训练所需的 PyTorch 张量

        Args:
            sample_data: sample 数据列表（可来自单个或多个 sample.json 合并）

        Returns:
            (inputs_tensor, targets_tensor, masks_tensor): 三个张量，已移到 GPU
        """
        # 性能优化：使用列表推导替代逐条 append，减少 Python 解释器开销
        inputs = [item["input"] for item in sample_data]
        targets = [item["output"] for item in sample_data]
        masks = [item["mask"] for item in sample_data]

        # 转换输入为模型所需的格式
        if self.input_converter is not None:
            inputs_tensor = self.input_converter(inputs)
        else:
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

        targets_tensor = torch.tensor(targets, dtype=torch.float32)  # (N, 225)
        masks_tensor = torch.tensor(masks, dtype=torch.float32)      # (N, 225)

        # 将张量移到 GPU（如果可用）
        inputs_tensor = inputs_tensor.to(self.device)
        targets_tensor = targets_tensor.to(self.device)
        masks_tensor = masks_tensor.to(self.device)

        # 注意：targets 已经是 Softmax 概率分布（由 Evaluator 生成）
        # 值域 (0, 1)，所有 mask=1 位置的概率和为 1
        # 不需要额外的归一化操作

        # mask 和 targets 都是 225 维，直接返回
        return inputs_tensor, targets_tensor, masks_tensor
