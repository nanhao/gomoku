"""SampleCreator 样本生成器模块

从 record.json 生成训练样本。

当前行为：
    - winner_path: 取整局中获胜方的每一步，output 采用**软标签**
      （第三十九次追加）——在模型自身 softmax 输出的基础上，把落子位
      最小增强为严格 top1，增量由其他 224 位等额汲取、clip 到 0、再归一化
      回 sum=1。
    - 默认启用 D4 增强（8 变换），可通过 ``augment=False`` 禁用
    - 支持步级过滤（变异步过滤），通过 ``move_filter`` 参数注入

注意：winning_move 模式已移除（意图 9）。
"""

import json
import os
from typing import Callable, List, Dict, Optional, Tuple


# ----------------------------------------------------------------------
# 第三十九次追加：winner_path 软标签抬升阈值
#
# EPS 用于保证落子位在归一化前严格超过当前 top1：
#     delta = p_max - p_move + EPS
# 取 1e-3 是设计评审后的取值（见 key_points.md 第 647 行、设计文档第三十九次追加）。
# 过小（如 1e-6）会在归一化后被浮点精度抹平；过大则让标签过于尖锐，
# 退化为近似 one-hot，失去"保留模型分布"的初衷。
# ----------------------------------------------------------------------
WINNER_PATH_TOP1_EPS = 1e-3


class SampleCreator:
    """样本生成器，从 record.json 生成训练样本

    仅支持 winner_path 模式（取获胜方每一步，软标签 output）。
    支持可选的步级过滤（变异步过滤）和 D4 数据增强。
    """

    def __init__(self, augment: bool = True):
        """
        初始化样本生成器

        Args:
            augment: 是否启用数据增强（D4 群 8 变换），默认 True
        """
        self.augment = augment

    def create(self, record_files: List[str],
               record_filter: Optional[Callable[[dict], bool]] = None,
               move_filter: Optional[Callable[[dict], bool]] = None) -> Tuple[List[Dict], dict]:
        """从 record.json 文件列表生成 sample 列表

        固定使用 winner_path 逻辑：取获胜方每一步，构造软标签 output。

        Args:
            record_files: record.json 文件路径列表
            record_filter: 可选的记录过滤函数。接收 record 字典，
                返回 True 保留、False 跳过。None 表示不过滤。
            move_filter: 可选的步级过滤函数。接收单个 history step dict
                （含 ``output`` 和 ``position`` 字段），
                返回 True 保留、False 跳过。None 表示不过滤。
                始终生效（温度采样始终启用）。

        Returns:
            (samples, stats) 二元组：
            - samples: sample 数据列表
            - stats: 步级过滤统计信息 dict，含以下键：
                - total_moves: 获胜方总步数
                - variant_moves: 变异步数（通过 move_filter 的步）
                - skipped_moves: 被 move_filter 跳过的步数
                - move_filter_active: 是否有任何对局启用了步级过滤
        """
        samples = []
        total_moves = 0
        variant_moves = 0
        skipped_moves = 0
        move_filter_active = False

        for file_path in record_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                record = json.load(f)

            # 公共过滤：跳过平局或未结束的对局
            winner = record.get("winner", 0)
            if winner == 0 or winner == 3:
                continue

            history = record.get("history", [])
            if not history:
                continue

            # 外部过滤：调用方注入的自定义过滤逻辑
            if record_filter is not None and not record_filter(record):
                continue

            # 固定使用 winner_path 逻辑：取获胜方每一步
            raw_samples, game_total, game_variant, game_skipped = \
                self._build_winner_path_samples(winner, history, move_filter)

            total_moves += game_total
            variant_moves += game_variant
            skipped_moves += game_skipped
            if move_filter is not None:
                move_filter_active = True

            # 可选 D4 增强（直接按 augment 属性决定）
            for raw in raw_samples:
                if self.augment:
                    samples.extend(self._augment_sample(raw))
                else:
                    samples.append(raw)

        stats = {
            "total_moves": total_moves,
            "variant_moves": variant_moves,
            "skipped_moves": skipped_moves,
            "move_filter_active": move_filter_active,
        }
        return samples, stats

    # ------------------------------------------------------------------
    # winner_path 样本构造
    # ------------------------------------------------------------------

    def _build_winner_path_samples(self, winner: int, history: list,
                                    move_filter: Optional[Callable[[dict], bool]] = None
                                    ) -> Tuple[List[Dict], int, int, int]:
        """winner_path 模式：取获胜方每一步

        - winner=1（黑胜）：获胜方在 history 中的偶数索引 [0, 2, 4, ...]
        - winner=2（白胜）：奇数索引 [1, 3, 5, ...]
        - 支持可选的步级过滤（move_filter）

        Args:
            winner: 胜者（1=黑，2=白）
            history: 对局历史步列表
            move_filter: 可选的步级过滤函数，None 表示不过滤

        Returns:
            (samples, total, variant, skipped) 四元组：
            - samples: 本局的样本列表（未增强）
            - total: 获胜方总步数
            - variant: 通过 move_filter 的步数
            - skipped: 被 move_filter 跳过的步数

        Note:
            output 构造采用第三十九次追加的**软标签**算法，详见
            :meth:`_build_winner_path_output`。
        """
        start_idx = 0 if winner == 1 else 1
        winner_indices = list(range(start_idx, len(history), 2))
        kept_indices = winner_indices

        total = len(kept_indices)
        variant = 0
        skipped = 0

        samples = []
        for i in kept_indices:
            step = history[i]

            # 步级过滤（始终生效，move_filter 传入时即执行）
            if move_filter is not None:
                if not move_filter(step):
                    skipped += 1
                    continue
            variant += 1

            # 软标签 output：基于当步模型概率分布构造
            output = self._build_winner_path_output(
                step["output"], step["position"]
            )
            # 构造样本，step_id 为首字段
            s = {}
            if "step_id" in step:
                s["step_id"] = step["step_id"]
            s["input"] = step["input"]
            s["output"] = output
            s["mask"] = [1] * 225
            samples.append(s)
        return samples, total, variant, skipped

    @staticmethod
    def _build_winner_path_output(model_output: list, position: list) -> list:
        """第三十九次追加：winner_path 软标签 output 构造

        算法（详见 design/sample_creator_design.md "第三十九次追加"）：
            1. p = 模型当步概率输出的拷贝（225 维，sum≈1）
            2. idx = position[0]*15 + position[1]
               - 若 idx 越界：output = p（原分布，不抬升）
            3. 若落子位已是"严格 top1"（唯一最大）：output = p（原样）
            4. 否则：
                 q[idx] = max(p) + EPS              (EPS = WINNER_PATH_TOP1_EPS)
                 q[j]   = p[j]       (j != idx)    (其余位置保持原始值)
                 output = q / sum(q)                (直接归一化)

        该算法保证：
            - output 非负且 sum 精确为 1
            - 落子位在归一化后仍是严格最大（严格 top1）
            - 非落子位之间的相对比例与原始分布完全一致
            - 缺省 p 为"退化全零"时也能工作（此时 output 退化为 one-hot）

        Args:
            model_output: 长度 225 的浮点列表，当步模型概率分布（predict 返回值）
            position: [row, col] 落子坐标

        Returns:
            长度 225 的浮点列表：软标签 output
        """
        # 原分布的拷贝；无论是否抬升都必须拷贝，避免外部修改污染原 record
        p = list(model_output)

        r, c = position[0], position[1]
        # 严格按 (r, c) 双维度各自在 [0, 15) 内判断越界；
        # 与硬标签版本保持一致的越界语义，避免"行合法列越界"被误写到错位。
        if not (0 <= r < 15 and 0 <= c < 15):
            # 越界兜底：直接返回模型原分布，不做任何抬升
            return p

        idx = r * 15 + c
        p_move = p[idx]
        p_max = max(p)

        # "严格 top1"判定：落子位等于最大值，且最大值在全表中仅出现一次
        if p_move == p_max and p.count(p_max) == 1:
            return p

        # ---- 抬升落子位 + 直接归一化 ----
        # 旧算法使用 per_share 均摊扣除，在极端分布下会 clip 清零大量低概率位置，
        # 导致落子位份额虚高，且破坏非落子位之间的原始比例关系。
        # 修复：去掉 per_share 扣除，仅抬升落子位后直接归一化。
        # 效果：落子位 = (p_max + EPS) / sum，比原冠军高 EPS 级别；
        #       所有非落子位的相对比例与原始分布完全一致。
        q = list(p)
        q[idx] = p_max + WINNER_PATH_TOP1_EPS

        # 归一化。q[idx] = p_max + EPS > 0，故 sum(q) > 0，除法安全
        s = sum(q)
        return [v / s for v in q]


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
            # 构造增强变体，step_id 为首字段
            aug = {}
            if "step_id" in sample:
                aug["step_id"] = sample["step_id"]
            aug["input"] = new_board + [player]
            aug["output"] = new_output
            aug["mask"] = new_mask
            augmented.append(aug)
        return augmented

    def create_from_round_dir(self, round_dir: str,
                              record_filter: Optional[Callable[[dict], bool]] = None,
                              move_filter: Optional[Callable[[dict], bool]] = None) -> Tuple[List[Dict], dict]:
        """从轮次目录自动发现所有 record.json 并生成 sample

        Args:
            round_dir: 轮次目录路径（含子目录如 1/record.json, 2/record.json）
            record_filter: 可选的记录过滤函数，透传给 create()
            move_filter: 可选的步级过滤函数，透传给 create()

        Returns:
            (samples, stats) 二元组，与 create() 一致
        """
        record_files = []
        if not os.path.exists(round_dir):
            return [], {"total_moves": 0, "variant_moves": 0,
                        "skipped_moves": 0, "move_filter_active": False}

        for entry in os.listdir(round_dir):
            entry_path = os.path.join(round_dir, entry)
            if os.path.isdir(entry_path):
                record_path = os.path.join(entry_path, "record.json")
                if os.path.exists(record_path):
                    record_files.append(record_path)

        return self.create(record_files, record_filter=record_filter,
                           move_filter=move_filter)
