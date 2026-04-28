"""样本数据加载与预处理模块

从 base_trainer.py 提取，统一负责从 sample.json 到 PyTorch 张量的全链路转换。
"""

import json
import os
from typing import List, Tuple, Callable, Optional

import torch


def compute_masked_l1_loss(estimated: List[float], sample: List[float], mask: List[int]) -> float:
    """纯 Python 实现的 masked L1 loss（供单元测试使用）

    计算 mask=1 的位置的平均绝对误差。

    Args:
        estimated: 模型预测的 reward 列表（225维）
        sample: sample.json 中的 reward 列表（225维）
        mask: mask 列表（225维）

    Returns:
        mask=1 位置的平均绝对误差；若 mask 全为 0 则返回 0.0
    """
    total_loss = 0.0
    mask_sum = 0
    for i in range(225):
        if mask[i] == 1:
            total_loss += abs(estimated[i] - sample[i])
            mask_sum += 1
    return total_loss / mask_sum if mask_sum > 0 else 0.0


def discover_all_data(data_dir: str) -> dict:
    """发现目录下所有训练数据源

    扫描 data_dir：
    1. 找到所有根目录下的 merge_*.json 文件
    2. 找到所有包含 sample.json 的直接子目录
    3. 返回二者列表

    Args:
        data_dir: 训练数据目录路径

    Returns:
        {
            "merge_files": List[str],    # merge_*.json 文件路径列表
            "sample_dirs": List[str]     # 包含 sample.json 的子目录列表
        }
    """
    merge_files = []
    sample_dirs = []

    if not os.path.exists(data_dir):
        return {"merge_files": [], "sample_dirs": []}

    try:
        entries = os.listdir(data_dir)
    except OSError:
        return {"merge_files": [], "sample_dirs": []}

    for entry in entries:
        entry_path = os.path.join(data_dir, entry)
        if os.path.isfile(entry_path) and entry.startswith("merge_") and entry.endswith(".json"):
            merge_files.append(entry_path)
        elif os.path.isdir(entry_path):
            sample_path = os.path.join(entry_path, "sample.json")
            if os.path.exists(sample_path):
                sample_dirs.append(entry_path)

    # 按文件名排序，保证结果稳定
    merge_files.sort()
    sample_dirs.sort(key=lambda p: os.path.basename(p))

    return {
        "merge_files": merge_files,
        "sample_dirs": sample_dirs,
    }


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

    def load_merge_file(self, merge_path: str) -> list:
        """加载合并后的样本文件

        与 load 方法行为一致，但语义上专用于加载 merge_*.json 文件。

        Args:
            merge_path: merge_*.json 文件路径

        Returns:
            sample 数据列表

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON 格式错误
        """
        with open(merge_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def discover_and_load(self, data_dir: str) -> list:
        """从目录发现并加载所有训练数据（merge 文件 + 个体 sample.json）

        扫描指定目录：
        1. 找到所有 merge_*.json 文件 → 加载
        2. 找到所有包含 sample.json 的子目录 → 加载
        3. 合并所有数据返回

        Args:
            data_dir: 训练数据目录路径

        Returns:
            合并后的 sample 数据列表
        """
        result = discover_all_data(data_dir)
        all_data = []

        # 加载所有 merge 文件
        for merge_file in result["merge_files"]:
            all_data.extend(self.load_merge_file(merge_file))

        # 加载所有个体 sample.json
        for sample_dir in result["sample_dirs"]:
            sample_path = os.path.join(sample_dir, "sample.json")
            all_data.extend(self.load(sample_path))

        return all_data

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
