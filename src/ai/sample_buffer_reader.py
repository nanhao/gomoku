"""样本缓冲区读取器模块

从 sample_buffer/ 目录读取样本缓冲区的 chunk JSON 文件，
提供随机不放回抽样功能，供训练脚本使用。
"""

import json
import os
import random
from typing import List


class SampleBufferReader:
    """样本缓冲区读取器

    从 sample_buffer/ 目录读取 manifest.json 和所有 chunk JSON 文件，
    将所有样本加载到内存中，提供随机不放回抽样。

    支持 refresh() 重新扫描缓冲区目录以加载新 chunk。
    """

    def __init__(self, buffer_dir: str):
        """初始化读取器，加载缓冲区中的所有样本

        Args:
            buffer_dir: 样本缓冲区目录绝对路径
        """
        self._buffer_dir: str = os.path.abspath(buffer_dir)
        self._all_samples: List[dict] = []
        self._remaining_indices: List[int] = []
        self._loaded_chunk_files: set = set()

        self._load_all()

    def random_sample(self, n: int) -> List[dict]:
        """不放回随机抽取 n 条样本

        从当前已加载的样本池中不放回随机抽取 n 条样本。
        若请求数超过剩余样本数，返回全部剩余样本。

        Args:
            n: 需要抽取的样本数

        Returns:
            抽取的样本列表，长度 ≤ n
        """
        if not self._remaining_indices:
            return []

        k = min(n, len(self._remaining_indices))
        chosen = random.sample(self._remaining_indices, k)

        result = [self._all_samples[i] for i in chosen]

        # 从剩余池中移除已抽取的索引
        chosen_set = set(chosen)
        self._remaining_indices = [i for i in self._remaining_indices if i not in chosen_set]

        return result

    def refresh(self) -> int:
        """重新扫描缓冲区目录，加载新 chunk

        仅加载尚未加载过的 chunk 文件。已抽取过的样本不会被重新加入抽样池；
        抽样池会用新加载的样本重新填充。

        Returns:
            本次新增的样本数
        """
        old_count = len(self._all_samples)
        self._load_all()
        new_count = len(self._all_samples) - old_count

        # 重置抽样池（包含所有样本，因为训练中不应重复使用旧样本）
        self._remaining_indices = list(range(len(self._all_samples)))

        return new_count

    def total_samples(self) -> int:
        """返回当前已加载样本总数

        Returns:
            已加载到内存的样本总数
        """
        return len(self._all_samples)

    # ---------- 内部方法 ----------

    def _load_all(self) -> None:
        """加载缓冲区中的所有 chunk 文件

        读取 manifest.json 获取 chunk 列表，逐个加载 chunk JSON 文件。
        损坏的 chunk 文件静默跳过。磁盘上不存在的 chunk 文件静默跳过。
        """
        manifest_path = os.path.join(self._buffer_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            return

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, IOError):
            return

        chunks = manifest.get("chunks", [])
        for chunk_meta in chunks:
            filename = chunk_meta.get("file", "")
            if not filename:
                continue

            # 跳过已加载的 chunk
            if filename in self._loaded_chunk_files:
                continue

            chunk_path = os.path.join(self._buffer_dir, filename)
            if not os.path.isfile(chunk_path):
                continue

            try:
                with open(chunk_path, "r", encoding="utf-8") as f:
                    chunk_samples = json.load(f)
                if isinstance(chunk_samples, list):
                    self._all_samples.extend(chunk_samples)
                    self._loaded_chunk_files.add(filename)
            except (json.JSONDecodeError, IOError):
                # 损坏的 chunk 文件静默跳过
                continue

        # 首次加载时初始化抽样池；refresh 后重建
        if len(self._remaining_indices) != len(self._all_samples):
            self._remaining_indices = list(range(len(self._all_samples)))
