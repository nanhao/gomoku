"""SampleCreator 样本生成器模块

支持多种模式从 record.json 生成训练样本。
目前实现模式：
    - winning_move: 取每局最后一步的 input 作为 sample input，
      最后一步 position 对应位置 output=1，其余=0，mask 全为 1
"""

import json
import os
from typing import List, Dict


class SampleCreator:
    """样本生成器，支持多种模式从 record.json 生成 sample"""

    def __init__(self, mode: str = "winning_move"):
        """
        初始化样本生成器

        Args:
            mode: 样本生成模式，目前仅支持 "winning_move"

        Raises:
            ValueError: 不支持的模式
        """
        if mode != "winning_move":
            raise ValueError(f"不支持的样本生成模式: '{mode}'，目前仅支持 'winning_move'")
        self.mode = mode

    def create(self, record_files: List[str]) -> List[Dict]:
        """从 record.json 文件列表生成 sample 列表

        Args:
            record_files: record.json 文件路径列表

        Returns:
            sample 数据列表，每个元素为 {"input": [...], "output": [...], "mask": [...]}
        """
        samples = []
        for file_path in record_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                record = json.load(f)

            # 跳过平局或未结束的对局
            winner = record.get("winner", 0)
            if winner == 0 or winner == 3:
                continue

            history = record.get("history", [])
            if not history:
                continue

            # 取最后一步
            last_step = history[-1]
            input_vector = last_step["input"]
            position = last_step["position"]

            # 构建 output：全 0，最后一步位置为 1
            output = [0.0] * 225
            idx = position[0] * 15 + position[1]
            if 0 <= idx < 225:
                output[idx] = 1.0

            # mask 全为 1
            mask = [1] * 225

            samples.append({
                "input": input_vector,
                "output": output,
                "mask": mask
            })

        return samples

    def create_from_round_dir(self, round_dir: str) -> List[Dict]:
        """从轮次目录自动发现所有 record.json 并生成 sample

        Args:
            round_dir: 轮次目录路径（含子目录如 1/record.json, 2/record.json）

        Returns:
            sample 数据列表
        """
        record_files = []
        if not os.path.exists(round_dir):
            return []

        for entry in os.listdir(round_dir):
            entry_path = os.path.join(round_dir, entry)
            if os.path.isdir(entry_path):
                record_path = os.path.join(entry_path, "record.json")
                if os.path.exists(record_path):
                    record_files.append(record_path)

        return self.create(record_files)
