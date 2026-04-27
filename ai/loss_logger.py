"""训练 Loss 日志记录器模块

将训练过程中的 loss 值记录到 JSON Lines 文件，
并支持生成 loss 曲线图。

支持通过依赖注入传入或禁用，实现灵活的日志配置与测试替换。
"""

import json
import os
import uuid
import warnings
from datetime import datetime

# 尝试导入 matplotlib，用于生成 loss 曲线图
# 若不可用则优雅降级，仅记录日志文件
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端，避免在无 GUI 环境报错
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

from ai.constants import LOSS_LOG_DIR


class LossLogger:
    """训练 Loss 日志记录器

    将训练过程中的 loss 值记录到 JSON Lines 文件，
    并支持生成 loss 曲线图。

    支持通过依赖注入传入或禁用，实现灵活的日志配置与测试替换。
    支持 session_dir 子目录隔离、语义化文件名及合并对比图。
    """

    def __init__(
        self,
        model_name: str,
        log_dir: str = None,
        session_dir: str = None,
        log_type: str = None,
        round_num: int = None,
        total_rounds: int = None,
        epochs: int = None,
        games: int = None,
        dense_log: bool = False
    ):
        """初始化日志记录器

        Args:
            model_name: 模型名称，如 "fc_v1"、"cnn_v1"
            log_dir: 日志目录，默认为 ai/ai_data/loss_logs
            session_dir: 训练会话子目录名，提供时日志存放在子目录下
            log_type: 日志层级标识，可选 "round_level" / "epoch_level"
            round_num: 当前轮次编号（从1开始）
            total_rounds: 总轮次数
            epochs: 该轮次训练的总 epoch 数
            games: 每轮对局数
            dense_log: 是否启用密集日志模式（仅 mini-batch 有效），默认 False
        """
        self.model_name = model_name
        self.log_dir = log_dir if log_dir is not None else LOSS_LOG_DIR
        self.session_dir = session_dir
        self.log_type = log_type
        self.round_num = round_num
        self.total_rounds = total_rounds
        self.epochs = epochs
        self.games = games
        self.dense_log = dense_log
        self._buffer = []  # 密集日志内存缓冲区

        # 确定实际日志目录（支持 session_dir 子目录隔离）
        if session_dir is not None:
            self.log_dir = os.path.join(self.log_dir, session_dir)

        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)

        # 根据 log_type 生成语义化文件名
        filename = self._generate_filename()
        self._log_path = os.path.join(self.log_dir, filename)

        # 预先创建空日志文件，确保 get_log_path 始终返回有效路径
        open(self._log_path, 'w', encoding='utf-8').close()

    def _generate_filename(self) -> str:
        """生成日志文件名

        Returns:
            语义化文件名（含层级、轮次信息）或旧格式 UUID 文件名
        """
        if self.log_type == "round_level":
            return "round_level.jsonl"
        elif self.log_type == "epoch_level" and self.round_num is not None:
            return f"round_{self.round_num}_epoch_level.jsonl"

        # 回退到旧格式（向后兼容）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return f"{self.model_name}_{timestamp}_{short_uuid}.jsonl"

    def _generate_title(self) -> str:
        """生成图表标题

        Returns:
            携带模型名、层级、轮次等语义信息的标题字符串
        """
        if self.log_type == "round_level":
            return (
                f"Loss Curve - {self.model_name} "
                f"(Round Level, {self.total_rounds} rounds, {self.games} games/round)"
            )
        elif self.log_type == "epoch_level" and self.round_num is not None:
            return (
                f"Loss Curve - {self.model_name} "
                f"(Round {self.round_num}/{self.total_rounds} Epoch Level, {self.epochs} epochs)"
            )

        # 回退到旧格式标题
        return f"Loss Curve - {self.model_name}"

    def log(self, epoch: int, loss: float, grad_norm: float = None) -> None:
        """记录一条 loss 日志

        Args:
            epoch: 当前 epoch 编号（从1开始）或轮次编号
            loss: 当前 loss 值
            grad_norm: 梯度 L2 范数，提供时一并记录
        """
        entry = {"epoch": epoch, "loss": loss}
        if grad_norm is not None:
            entry["grad_norm"] = grad_norm
        with open(self._log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")

    def log_dense(self, step: int, loss: float, grad_norm: float) -> None:
        """记录一条密集日志（存入内存缓冲区，不直接写文件）

        用于 mini-batch 模式下记录每个 batch 的 loss 和梯度范数。
        数据先缓存在内存中，调用 flush() 时才批量写入磁盘。

        Args:
            step: 全局步数（batch 序号）
            loss: 当前 batch 的 loss 值
            grad_norm: 当前 batch 的梯度 L2 范数
        """
        self._buffer.append({"step": step, "loss": loss, "grad_norm": grad_norm})

    def flush(self) -> None:
        """将内存缓冲区中的所有日志批量写入文件

        应在每个 epoch 结束后或训练结束时调用，确保数据落盘。
        """
        if self._buffer:
            with open(self._log_path, 'a', encoding='utf-8') as f:
                for entry in self._buffer:
                    f.write(json.dumps(entry) + "\n")
            self._buffer.clear()

    def plot(self, output_path: str = None) -> str:
        """根据已记录的日志生成 loss 曲线图

        如果日志中包含 grad_norm 字段，绘制双 Y 轴图（左轴 Loss，右轴 Grad Norm）。
        数据点超过 5000 时自动降采样，防止图表过于密集。

        Args:
            output_path: 图片保存路径，默认为日志目录下同名 .png

        Returns:
            图片文件路径；若无可记录数据或 matplotlib 不可用则返回 None
        """
        # 读取所有日志记录
        entries = self._read_entries()
        if not entries:
            return None

        # matplotlib 不可用时优雅降级（导入时或运行时均处理）
        if not _MATPLOTLIB_AVAILABLE:
            warnings.warn(
                "matplotlib 不可用，跳过 loss 曲线图生成。日志文件不受影响。",
                RuntimeWarning
            )
            return None

        # 确定输出路径
        if output_path is None:
            output_path = self._log_path.replace(".jsonl", ".png")

        # 判断是否包含 grad_norm（密集日志模式）
        has_grad_norm = "grad_norm" in entries[0]

        try:
            if has_grad_norm:
                # 双 Y 轴图（Loss + Grad Norm）
                entries = self._downsample(entries, max_points=5000)
                losses = [entry["loss"] for entry in entries]
                grad_norms = [entry["grad_norm"] for entry in entries]

                # 判断是否有 step 字段（密集模式）还是 epoch 字段（稀疏模式）
                has_step = "step" in entries[0]
                x_values = [entry["step" if has_step else "epoch"] for entry in entries]
                x_label = "Step" if has_step else "Epoch"

                fig, ax1 = plt.subplots(figsize=(12, 7))
                ax2 = ax1.twinx()

                ax1.plot(x_values, losses, 'b-', label='Loss', linewidth=1)
                ax2.plot(x_values, grad_norms, 'r-', label='Grad Norm', linewidth=1, alpha=0.7)

                ax1.set_xlabel(x_label)
                ax1.set_ylabel("Loss", color='b')
                ax2.set_ylabel("Gradient Norm", color='r')
                ax1.tick_params(axis='y', labelcolor='b')
                ax2.tick_params(axis='y', labelcolor='r')

                # 合并图例
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

                ax1.grid(True, linestyle='--', alpha=0.7)
                plt.title(self._generate_title())
                plt.tight_layout()
                plt.savefig(output_path, dpi=150)
                plt.close()
            else:
                # 稀疏模式：单 Y 轴图（向后兼容）
                epochs = [entry["epoch"] for entry in entries]
                losses = [entry["loss"] for entry in entries]

                plt.figure(figsize=(10, 6))
                plt.plot(epochs, losses, marker='o', linestyle='-', linewidth=1.5, markersize=3)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(self._generate_title())
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(output_path, dpi=150)
                plt.close()

            return output_path
        except Exception:
            # 运行时 matplotlib 异常也优雅降级，不影响训练流程
            warnings.warn(
                "matplotlib 绘图失败，跳过 loss 曲线图生成。日志文件不受影响。",
                RuntimeWarning
            )
            return None

    def get_log_path(self) -> str:
        """返回当前日志文件路径"""
        return self._log_path

    def clear(self) -> None:
        """清空当前日志文件（新训练开始时调用）"""
        open(self._log_path, 'w', encoding='utf-8').close()
        self._buffer.clear()

    def _read_entries(self) -> list:
        """读取日志文件中的所有记录

        Returns:
            日志记录列表，每个元素为 {"epoch": int, "loss": float}
            或 {"step": int, "loss": float, "grad_norm": float}
        """
        entries = []
        if not os.path.exists(self._log_path):
            return entries

        with open(self._log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 忽略格式错误的行，保证健壮性
                        continue

        return entries

    def _downsample(self, entries: list, max_points: int = 5000) -> list:
        """降采样：超过 max_points 时合并相邻点

        通过滑动窗口取平均的方式减少数据点数量，防止图表过于密集。

        Args:
            entries: 原始日志记录列表
            max_points: 最大允许的数据点数，默认 5000

        Returns:
            降采样后的记录列表
        """
        if len(entries) <= max_points:
            return entries

        window = max(1, len(entries) // max_points)
        result = []
        for i in range(0, len(entries), window):
            chunk = entries[i:i + window]
            avg_step = sum(e["step"] for e in chunk) / len(chunk)
            avg_loss = sum(e["loss"] for e in chunk) / len(chunk)
            avg_grad = sum(e["grad_norm"] for e in chunk) / len(chunk)
            result.append({"step": avg_step, "loss": avg_loss, "grad_norm": avg_grad})
        return result

    @classmethod
    def plot_comparison(
        cls,
        session_dir: str,
        model_name: str = None,
        log_dir: str = None,
        output_path: str = None
    ) -> str:
        """将 session_dir 下所有 epoch_level 日志合并绘制到一张对比图

        每轮一条不同颜色的折线，统一 Y 轴范围，带图例标识。

        Args:
            session_dir: 训练会话子目录名
            model_name: 模型名称，用于图表标题
            log_dir: 日志根目录，默认为 ai/ai_data/loss_logs
            output_path: 图片保存路径，默认保存为 comparison.png

        Returns:
            生成的图片路径；无数据或 matplotlib 不可用时返回 None
        """
        if not _MATPLOTLIB_AVAILABLE:
            warnings.warn(
                "matplotlib 不可用，跳过合并对比图生成。",
                RuntimeWarning
            )
            return None

        base_dir = log_dir if log_dir is not None else LOSS_LOG_DIR
        session_path = os.path.join(base_dir, session_dir)

        if not os.path.isdir(session_path):
            return None

        # 收集所有 epoch_level 日志数据
        all_data = []
        for fname in sorted(os.listdir(session_path)):
            if fname.startswith("round_") and fname.endswith("_epoch_level.jsonl"):
                try:
                    round_num = int(fname.split("_")[1])
                except (IndexError, ValueError):
                    continue

                fpath = os.path.join(session_path, fname)
                entries = []
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

                if entries:
                    all_data.append((round_num, entries))

        if not all_data:
            return None

        try:
            # 绘制合并对比图
            plt.figure(figsize=(12, 7))
            xlabel = "Epoch"
            for round_num, entries in sorted(all_data, key=lambda x: x[0]):
                # 兼容稀疏日志（epoch）和密集日志（step）两种格式
                if entries and "step" in entries[0]:
                    xs = [entry["step"] for entry in entries]
                    xlabel = "Step"
                else:
                    xs = [entry["epoch"] for entry in entries]
                losses = [entry["loss"] for entry in entries]
                plt.plot(
                    xs, losses,
                    marker='o', linestyle='-', linewidth=1.5, markersize=3,
                    label=f"Round {round_num}"
                )

            plt.xlabel(xlabel)
            plt.ylabel("Loss")
            plt.title(f"Loss Comparison - {model_name or 'Model'}")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            if output_path is None:
                output_path = os.path.join(session_path, "comparison.png")
            plt.savefig(output_path, dpi=150)
            plt.close()
            return output_path
        except Exception:
            warnings.warn(
                "matplotlib 合并绘图失败，跳过对比图生成。",
                RuntimeWarning
            )
            return None
