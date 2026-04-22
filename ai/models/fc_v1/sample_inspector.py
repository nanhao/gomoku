"""FC v1 样本检视器（SampleInspector）

读取 sample.json，逐步调用模型 predict，将模型输出与 sample output 做对比并格式化打印。
注意：此文件命名为 sample_inspector.py 以避免与 Python 标准库 inspect 模块命名冲突。

输出格式适配：Evaluator 现输出 Softmax 概率分布（0~1），而非原始 reward 值（±1e6）。
- 已占位位置：概率为 0.0（被 Softmax mask 排除）
- 空位位置：概率为 0~1 之间的浮点数，总和为 1
- 胜利位置：概率最高的位置
"""

import json
from typing import List, Optional

from .model import FCv1Model

# Softmax 概率阈值：概率高于此值的位置视为胜利位置
# 典型的胜利位置概率 > 0.5，远高于其他空位
_WIN_PROB_THRESHOLD = 0.5


class SampleInspector:
    """FC v1 样本检视器

    读取 sample.json，逐步将每步的 input 喂给模型 predict，
    与 sample 中的 output（target）做逐步对比，输出差异报告。
    """

    def __init__(self, model: FCv1Model):
        """初始化检视器

        Args:
            model: 已加载的 FCv1Model 实例
        """
        self.model = model

    def inspect(self, sample_path: str) -> List[dict]:
        """读取 sample.json，对每步调用 predict，返回每步的对比结果列表。

        Args:
            sample_path: sample.json 文件路径

        Returns:
            每步的对比结果列表，每个元素是一个 dict，结构见设计文档 section 12.2

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON 格式错误
        """
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total = len(data)
        results = []

        for step_i, item in enumerate(data):
            inp = item["input"]      # 226 维
            target = item["output"]  # 225 维，Softmax 概率分布（0~1）
            board = inp[:225]        # 前 225 位是棋盘状态
            player = inp[225]        # 最后 1 位是当前落子方

            # 统计已占位和空位数量
            occupied_count = sum(1 for v in board if v > 0)
            empty_count = 225 - occupied_count

            # 模型推理
            model_output = self.model.predict(inp)  # 225 维概率列表

            # 计算 L1 loss（Softmax 格式下直接比较概率值）
            loss = self._compute_loss(model_output, target)

            # 查找胜利位置：Softmax 格式下，概率最高的位置为胜利位置
            # 仅在最后一步检测（与旧格式行为一致）
            win_pos: Optional[int] = None
            win_model_output: Optional[float] = None
            if step_i == total - 1 and target:
                # 找到概率最高的位置
                max_prob = max(target)
                if max_prob > _WIN_PROB_THRESHOLD:
                    win_pos = target.index(max_prob)
                    win_model_output = float(model_output[win_pos])

            results.append({
                "step": step_i,
                "is_last": (step_i == total - 1),
                "player": int(player),
                "occupied_count": occupied_count,
                "empty_count": empty_count,
                "board": board,  # 保存棋盘状态供 format_step 使用
                "model_output": model_output,
                "target": target,
                "loss": loss,
                "win_pos": win_pos,
                "win_model_output": win_model_output,
            })

        return results

    def format_step(self, step_result: dict) -> str:
        """将单步对比结果格式化为可打印字符串。

        Args:
            step_result: inspect() 返回的单步结果字典

        Returns:
            格式化字符串
        """
        step = step_result["step"]
        is_last = step_result["is_last"]
        player = step_result["player"]
        occupied_count = step_result["occupied_count"]
        empty_count = step_result["empty_count"]
        board = step_result["board"]
        model_output = step_result["model_output"]
        target = step_result["target"]
        loss = step_result["loss"]
        win_pos = step_result["win_pos"]
        win_model_output = step_result["win_model_output"]

        player_str = "黑" if player == 1 else "白"
        last_tag = " [最后一步]" if is_last else ""

        lines = []
        lines.append(
            f"Step {step}{last_tag} ({occupied_count}子/{empty_count}空, "
            f"落子方: {player_str}, loss={loss:.4f}):"
        )

        # 已占位位置统计（Softmax 格式下，已占位位置概率为 0.0）
        occ_model_vals = [model_output[i] for i in range(225) if board[i] > 0]
        occ_target_vals = [target[i] for i in range(225) if board[i] > 0]
        if occ_model_vals:
            occ_model_mean = sum(occ_model_vals) / len(occ_model_vals)
            occ_target_mean = sum(occ_target_vals) / len(occ_target_vals)
            lines.append(
                f"  已占位({occupied_count}): "
                f"模型输出均值={occ_model_mean:.6f}, "
                f"target均值={occ_target_mean:.6f}"
            )

        # 空位统计（board 上为 0 的位置）
        empty_model_vals = [model_output[i] for i in range(225) if board[i] == 0]
        empty_target_vals = [target[i] for i in range(225) if board[i] == 0]
        if empty_model_vals:
            empty_model_mean = sum(empty_model_vals) / len(empty_model_vals)
            empty_target_mean = sum(empty_target_vals) / len(empty_target_vals)
            lines.append(
                f"  空位({empty_count}): "
                f"模型输出均值={empty_model_mean:.6f}, "
                f"target均值={empty_target_mean:.6f}"
            )

        # 胜利位置详情
        if win_pos is not None:
            row, col = win_pos // 15, win_pos % 15
            lines.append(
                f"  胜利位置: idx={win_pos} (row={row},col={col}), "
                f"概率={target[win_pos]:.6f}, "
                f"模型输出={win_model_output:.6f}"
            )
        else:
            lines.append("  胜利位置: 无")

        return "\n".join(lines)

    def _compute_loss(self, model_output: List[float], target: List[float]) -> float:
        """计算单步 L1 loss（Softmax 概率格式）。

        Softmax 概率范围 0~1，直接比较模型输出与 target 的 L1 距离。
        loss = mean(|model_output[i] - target[i]|) for all i

        Args:
            model_output: 模型输出的 225 维概率列表
            target: sample.json 中的 output（225维，Softmax概率分布）

        Returns:
            平均 L1 损失
        """
        total = 0.0
        for m, t in zip(model_output, target):
            total += abs(m - t)
        return total / len(model_output)
