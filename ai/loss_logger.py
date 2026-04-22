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
        games: int = None
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
        """
        self.model_name = model_name
        self.log_dir = log_dir if log_dir is not None else LOSS_LOG_DIR
        self.session_dir = session_dir
        self.log_type = log_type
        self.round_num = round_num
        self.total_rounds = total_rounds
        self.epochs = epochs
        self.games = games

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

    def log(self, epoch: int, loss: float) -> None:
        """记录一条 loss 日志

        Args:
            epoch: 当前 epoch 编号（从1开始）或轮次编号
            loss: 当前 loss 值
        """
        entry = {"epoch": epoch, "loss": loss}
        with open(self._log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")

    def plot(self, output_path: str = None) -> str:
        """根据已记录的日志生成 loss 曲线图

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

        # 提取 epoch 和 loss 数据
        epochs = [entry["epoch"] for entry in entries]
        losses = [entry["loss"] for entry in entries]

        try:
            # 绘制曲线图
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

    def _read_entries(self) -> list:
        """读取日志文件中的所有记录

        Returns:
            日志记录列表，每个元素为 {"epoch": int, "loss": float}
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
            for round_num, entries in sorted(all_data, key=lambda x: x[0]):
                epochs = [entry["epoch"] for entry in entries]
                losses = [entry["loss"] for entry in entries]
                plt.plot(
                    epochs, losses,
                    marker='o', linestyle='-', linewidth=1.5, markersize=3,
                    label=f"Round {round_num}"
                )

            plt.xlabel("Epoch")
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
