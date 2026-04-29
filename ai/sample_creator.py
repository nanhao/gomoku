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

    def __init__(self, mode: str = "winning_move", augment: bool = True):
        """
        初始化样本生成器

        Args:
            mode: 样本生成模式，目前仅支持 "winning_move"
            augment: 是否启用数据增强（D4 群 8 变换），默认 True

        Raises:
            ValueError: 不支持的模式
        """
        if mode != "winning_move":
            raise ValueError(f"不支持的样本生成模式: '{mode}'，目前仅支持 'winning_move'")
        self.mode = mode
        self.augment = augment

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

            sample = {
                "input": input_vector,
                "output": output,
                "mask": mask
            }

            if self.augment:
                samples.extend(self._augment_sample(sample))
            else:
                samples.append(sample)

        return samples

    @staticmethod
    def _transform_index(idx: int, tid: int) -> int:
        """对 15×15 棋盘的一维索引进行 D4 群变换

        Args:
            idx: 原始一维索引，idx = r * 15 + c，0 ≤ r, c < 15
            tid: 变换编号，0~7
                0: 恒等 (r, c)
                1: 顺时针90° (c, 14-r)
                2: 180° (14-r, 14-c)
                3: 顺时针270° (14-c, r)
                4: 左右镜像 (r, 14-c)
                5: 上下镜像 (14-r, c)
                6: 主对角线翻转 (c, r)
                7: 反对角线翻转 (14-c, 14-r)

        Returns:
            变换后的一维索引
        """
        r, c = divmod(idx, 15)
        if tid == 0:
            r2, c2 = r, c
        elif tid == 1:
            r2, c2 = c, 14 - r
        elif tid == 2:
            r2, c2 = 14 - r, 14 - c
        elif tid == 3:
            r2, c2 = 14 - c, r
        elif tid == 4:
            r2, c2 = r, 14 - c
        elif tid == 5:
            r2, c2 = 14 - r, c
        elif tid == 6:
            r2, c2 = c, r
        elif tid == 7:
            r2, c2 = 14 - c, 14 - r
        else:
            raise ValueError(f"不支持的变换编号: {tid}，仅支持 0~7")
        return r2 * 15 + c2

    @staticmethod
    def _apply_transform(vector_225: list, tid: int) -> list:
        """对 225 维向量应用 D4 群变换

        Args:
            vector_225: 长度为 225 的列表，表示 15×15 棋盘展平
            tid: 变换编号，0~7

        Returns:
            变换后的 225 维列表
        """
        result = [0.0] * 225
        for old_idx in range(225):
            new_idx = SampleCreator._transform_index(old_idx, tid)
            result[new_idx] = vector_225[old_idx]
        return result

    @staticmethod
    def _augment_sample(sample: dict) -> list:
        """对单个 sample 应用 D4 群全部 8 种变换，生成 8 个变体

        Args:
            sample: 原始样本字典，含 "input"(226维)、"output"(225维)、"mask"(225维)

        Returns:
            8 个变体样本字典的列表，第 0 个为恒等变换（与原始一致）
        """
        board = sample["input"][:225]
        player = sample["input"][225]
        output = sample["output"]
        mask = sample["mask"]

        augmented = []
        for tid in range(8):
            new_board = SampleCreator._apply_transform(board, tid)
            new_output = SampleCreator._apply_transform(output, tid)
            new_mask = SampleCreator._apply_transform(mask, tid)
            augmented.append({
                "input": new_board + [player],
                "output": new_output,
                "mask": new_mask
            })
        return augmented

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
