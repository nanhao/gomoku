"""SampleCreator 样本生成器模块

支持多种模式从 record.json 生成训练样本。
目前实现模式：
    - winning_move: 取每局最后一步的 input 作为 sample input，
      最后一步 position 对应位置 output=1，其余=0，mask 全为 1
    - winner_path:  取整局中获胜方的每一步，每一步的 input/mask
      构造同 winning_move；output 采用**软标签**
      （第三十九次追加）——在模型自身 softmax 输出的基础上，把落子位
      最小增强为严格 top1，增量由其他 224 位等额汲取、clip 到 0、再归一化
      回 sum=1。此模式**强制不做** D4 增强，详见
      design/sample_creator_design.md "第三十八/三十九次追加"。
"""

import json
import os
from typing import Callable, List, Dict


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
    """样本生成器，支持多种模式从 record.json 生成 sample"""

    # 支持的模式白名单
    SUPPORTED_MODES = ("winning_move", "winner_path")

    def __init__(self, mode: str = "winning_move", augment: bool = True):
        """
        初始化样本生成器

        Args:
            mode: 样本生成模式，取值范围见 ``SUPPORTED_MODES``
            augment: 是否启用数据增强（D4 群 8 变换），默认 True。
                     注意：当 mode="winner_path" 时会强制禁用增强，
                     以避免数据量失控（该模式本身已每局产出多个 sample）。

        Raises:
            ValueError: mode 不在支持列表内
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"不支持的样本生成模式: '{mode}'，"
                f"目前支持: {self.SUPPORTED_MODES}"
            )
        self.mode = mode
        # winner_path 模式强制禁用增强（设计决策，详见设计文档）
        self.augment = augment and mode != "winner_path"

    def create(self, record_files: List[str],
               record_filter: Callable[[dict], bool] = None) -> List[Dict]:
        """从 record.json 文件列表生成 sample 列表

        按 ``self.mode`` 分派到对应的处理路径：
        - ``winning_move``：每局产出 1 条（取最后一步），可选 D4 增强
        - ``winner_path``：每局产出 ``获胜方步数`` 条，不增强

        Args:
            record_files: record.json 文件路径列表
            record_filter: 可选的记录过滤函数。接收 record 字典，
                返回 True 保留、False 跳过。None 表示不过滤。

        Returns:
            sample 数据列表，每个元素为 {"input": [...], "output": [...], "mask": [...]}
        """
        samples = []
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
            # 注意：此过滤在"平局/空history跳过"之后执行，
            # 即使 record_filter 保留某对局，若其 history 为空仍会被跳过
            if record_filter is not None and not record_filter(record):
                continue

            # 按 mode 分派：每种模式返回"本局待产出的 sample 原型列表"（未增强）
            if self.mode == "winning_move":
                raw_samples = self._build_winning_move_samples(history)
            elif self.mode == "winner_path":
                raw_samples = self._build_winner_path_samples(winner, history)
            else:
                # SUPPORTED_MODES 在 __init__ 已拦截；此分支理论不可达，
                # 万一未来新增模式而忘记在此分派，立即暴露。
                raise AssertionError(f"unreachable: mode={self.mode!r}")

            # 可选 D4 增强（winner_path 已在 __init__ 中强制禁用）
            for raw in raw_samples:
                if self.augment:
                    samples.extend(self._augment_sample(raw))
                else:
                    samples.append(raw)

        return samples

    # ------------------------------------------------------------------
    # 各模式的样本构造
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sample(input_vector: list, position: list, step_id: str = None) -> Dict:
        """构造单个 sample：output 落子位=1.0（行/列任一越界则保持全 0），mask 全 1

        Args:
            input_vector: 226 维棋盘+玩家向量
            position: [x, y] 落子坐标
            step_id: 全局唯一步标识，从 record.json 透传；None 时不写入（向后兼容）

        Returns:
            sample 字典

        Note:
            严格按 (r, c) 双维度各自在 [0, 15) 内判断越界，
            而不是只看一维索引 ``r*15+c`` 是否在 [0, 225) 内——
            后者会把"行合法列越界"（如 [0, 20]）的 idx=20 误写到 (1, 5)。
        """
        output = [0.0] * 225
        r, c = position[0], position[1]
        if 0 <= r < 15 and 0 <= c < 15:
            output[r * 15 + c] = 1.0
        mask = [1] * 225
        sample = {}
        # step_id 首字段：仅当 record.json 中含 step_id 时写入样本
        if step_id is not None:
            sample["step_id"] = step_id
        sample["input"] = input_vector
        sample["output"] = output
        sample["mask"] = mask
        return sample

    def _build_winning_move_samples(self, history: list) -> List[Dict]:
        """winning_move 模式：仅取最后一步"""
        last_step = history[-1]
        step_id = last_step.get("step_id")
        return [self._build_sample(last_step["input"], last_step["position"], step_id)]

    def _build_winner_path_samples(self, winner: int, history: list) -> List[Dict]:
        """winner_path 模式：取获胜方每一步

        - winner=1（黑胜）：获胜方在 history 中的偶数索引 [0, 2, 4, ...]
        - winner=2（白胜）：奇数索引 [1, 3, 5, ...]
        - 获胜方步索引：winner=1 → [0, 2, 4, ...]；winner=2 → [1, 3, 5, ...]

        Note:
            output 构造采用第三十九次追加的**软标签**算法，详见
            :meth:`_build_winner_path_output`。
        """
        start_idx = 0 if winner == 1 else 1
        winner_indices = list(range(start_idx, len(history), 2))
        kept_indices = winner_indices

        samples = []
        for i in kept_indices:
            step = history[i]
            # 软标签 output：基于当步模型 softmax 输出构造
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
        return samples

    @staticmethod
    def _build_winner_path_output(model_output: list, position: list) -> list:
        """第三十九次追加：winner_path 软标签 output 构造

        算法（详见 design/sample_creator_design.md "第三十九次追加"）：
            1. p = 模型当步 softmax 输出的拷贝（225 维，sum≈1）
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
            model_output: 长度 225 的浮点列表，当步模型 softmax 输出
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
                              record_filter: Callable[[dict], bool] = None) -> List[Dict]:
        """从轮次目录自动发现所有 record.json 并生成 sample

        Args:
            round_dir: 轮次目录路径（含子目录如 1/record.json, 2/record.json）
            record_filter: 可选的记录过滤函数，透传给 create()

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

        return self.create(record_files, record_filter=record_filter)
