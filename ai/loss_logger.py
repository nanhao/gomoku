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
    """

    def __init__(self, model_name: str, log_dir: str = None):
        """初始化日志记录器

        Args:
            model_name: 模型名称，如 "fc_v1"、"cnn_v1"
            log_dir: 日志目录，默认为 ai/ai_data/loss_logs
        """
        self.model_name = model_name
        self.log_dir = log_dir if log_dir is not None else LOSS_LOG_DIR

        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)

        # 生成唯一日志文件名：{model_name}_{timestamp}_{short_uuid}.jsonl
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        filename = f"{model_name}_{timestamp}_{short_uuid}.jsonl"
        self._log_path = os.path.join(self.log_dir, filename)

        # 预先创建空日志文件，确保 get_log_path 始终返回有效路径
        open(self._log_path, 'w', encoding='utf-8').close()

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
            plt.title(f"Loss Curve - {self.model_name}")
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
