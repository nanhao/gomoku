"""样本数据加载与预处理模块

从 base_trainer.py 提取，统一负责从 sample.json 到 PyTorch 张量的全链路转换。
"""

import math
import json
import os
from typing import List, Tuple, Callable, Optional
import random

import torch


def compute_masked_kl_loss(log_estimated: List[float], target: List[float], mask: List[int]) -> float:
    """纯 Python 实现的 masked KL 散度 loss（供单元测试使用）

    计算 mask=1 位置的平均 KL 散度。
    KL 散度公式: Σ target[i] * (log(target[i]) - log_estimated[i]) / count

    **注意**: log_estimated 参数应为网络 LogSoftmax 输出的 log-probability（值 <= 0），
    而非 Softmax 概率值（值 > 0）。误传概率值会导致计算结果完全错误。

    Args:
        log_estimated: 模型输出的 log-probability 列表（225维），值域 (-inf, 0]
        target: sample.json 中的概率分布列表（225维），值域 [0, 1]
        mask: mask 列表（225维），1 表示空位，0 表示已占位

    Returns:
        mask=1 位置的平均 KL 散度；若 mask 全为 0 则返回 0.0

    Raises:
        ValueError: log_estimated 中存在 > 0 的元素（应为 log-probability 而非概率值）
    """
    # 运行时防御：检查 log_estimated 是否误传概率值
    for val in log_estimated:
        if val > 0:
            raise ValueError(
                "log_estimated 中存在 > 0 的元素（值={val}），"
                "应传入 log-probability 而非概率值。"
                "请确保传入的是网络 LogSoftmax 输出。"
            )

    total_loss = 0.0
    mask_sum = 0
    for i in range(225):
        if mask[i] == 1:
            # target[i]==0 时跳过 log(0) 计算，该位置贡献为 0
            if target[i] > 0:
                total_loss += target[i] * (math.log(target[i]) - log_estimated[i])
            mask_sum += 1
    return total_loss / mask_sum if mask_sum > 0 else 0.0


def discover_all_data(data_dir: str) -> list:
    """递归发现目录下所有训练数据文件

    递归扫描 data_dir，找到所有 .json 文件并返回排序后的路径列表。
    不区分文件名前缀（merge_、sample 等均加载），只要以 .json 结尾即视为训练数据。

    Args:
        data_dir: 训练数据目录路径

    Returns:
        排序后的 .json 文件路径列表
    """
    json_files = []

    if not os.path.exists(data_dir):
        return []

    try:
        for root, _dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".json"):
                    json_files.append(os.path.join(root, f))
    except OSError:
        return []

    # 按文件路径排序，保证结果稳定
    json_files.sort()

    return json_files


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

    def load_and_split(
        self,
        sample_dirs: List[str],
        val_ratio: float = 0.0,
        seed: int = 42
    ) -> Tuple[list, list]:
        """加载多个 sample.json 并按比例分割为训练集和验证集

        在样本级别进行随机分割，使用 random.Random(seed) 确保结果可复现。
        val_ratio=0 时所有数据归训练集，验证集为空。

        Args:
            sample_dirs: 包含 sample.json 的目录路径列表
            val_ratio: 验证集比例，默认 0.0（不分割）
            seed: 随机种子，确保分割可复现，默认 42

        Returns:
            (train_data, val_data): 训练集和验证集的 sample 数据列表

        Raises:
            ValueError: val_ratio < 0 或 val_ratio >= 1
            FileNotFoundError: 某个目录下不存在 sample.json
        """
        if val_ratio < 0 or val_ratio >= 1:
            raise ValueError(f"val_ratio 必须在 [0, 1) 范围内，当前值为 {val_ratio}")

        all_sample_data = self.load_multiple(sample_dirs)
        n_total = len(all_sample_data)

        # val_ratio=0 时直接返回全部训练集
        if val_ratio == 0 or n_total == 0:
            return all_sample_data, []

        # 使用独立 Random 实例确保可复现，不影响全局随机状态
        rng = random.Random(seed)
        indices = list(range(n_total))
        rng.shuffle(indices)

        n_val = max(1, int(n_total * val_ratio))
        # 边界保护：至少留一个样本给训练集
        if n_val >= n_total:
            n_val = n_total - 1
        if n_val < 0:
            n_val = 0

        val_indices = set(indices[:n_val])
        train_data = [all_sample_data[i] for i in range(n_total) if i not in val_indices]
        val_data = [all_sample_data[i] for i in range(n_total) if i in val_indices]

        return train_data, val_data

    def load_merge_file(self, merge_path: str) -> list:
        """加载合并后的样本文件

        与 load 方法行为一致，但语义上专用于加载 merge_*.json 文件。
        实际逻辑委托给 load 方法，避免重复实现。

        Args:
            merge_path: merge_*.json 文件路径

        Returns:
            sample 数据列表

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON 格式错误
        """
        return self.load(merge_path)

    def discover_and_load(self, data_dir: str) -> list:
        """从目录递归发现并加载所有训练数据

        递归扫描指定目录，找到所有 .json 文件并加载合并。
        不区分文件名前缀，所有 .json 文件统一加载。

        Args:
            data_dir: 训练数据目录路径

        Returns:
            合并后的 sample 数据列表
        """
        json_files = discover_all_data(data_dir)
        all_data = []

        # 加载所有发现的 .json 文件
        for json_file in json_files:
            all_data.extend(self.load(json_file))

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
